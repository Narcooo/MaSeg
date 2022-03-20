import os
import argparse
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

# from dataset.AIRS import DataSet
from dataset.LoveDA import DataSet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from segmentors import Segmentor
import transforms
def create_model(num_classes, pretrain=False):
    model = Segmentor(num_classes)

    if pretrain:
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_file = "results/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    train_images_path = args.data_path + 'Train/'
    val_images_path = args.data_path + 'Val/'
    img_size = 512
    data_transform = {
        "train": transforms.Compose([
                                     # T.RandomCrop(img_size,fill=1),
                                     transforms.ToTensor(),
                                     transforms.RandomCrop(img_size, mask_background_fill=0),
                                     transforms.RandomHorizontalFlip(),

                                     transforms.Normalize(mean=(123.675, 116.28, 103.53),
                           std=(58.395, 57.12, 57.375))
        ]),
                                     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.CenterCrop(512),

                                   transforms.Normalize(mean=(123.675, 116.28, 103.53),
                                 std=(58.395, 57.12, 57.375))
        ])}
                                   # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    # 实例化训练数据集
    train_dataset = DataSet(data_root=train_images_path,
                              transforms=data_transform["train"])

    # 实例化验证数据集
    val_dataset = DataSet(data_root=val_images_path,
                            transforms=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn
                                             )

    #model = create_model(num_classes=args.num_classes).to(device)
    model = Segmentor(num_classes=args.num_classes).to(device)
    print(get_parameter_number(model))
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        for k,v in weights_dict.items():
            if "head" in k:
                del weights_dict[k]
        del weights_dict['state_dict']['decode_head.conv_seg.weight']
        del weights_dict['state_dict']['decode_head.conv_seg.bias']
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict['state_dict'], strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.01)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    start_time = time.time()
    for epoch in range(args.epochs):
        # train
        mean_loss, lr = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              num_classes=args.num_classes)
        confmat = evaluate(model=model, data_loader=val_loader, device=device, num_classes=args.num_classes)
        val_info = str(confmat)
        print(val_info)
        tags = ["mean_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        # tb_writer.add_scalars(tags[2], np.array(confmat.mat.cpu()), epoch)
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        torch.save(save_file, "weights/"+args.save_path +"model-{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    # torch.save(model.state_dict(), "weights/"+args.save_path +"model-{}.pth".format(epoch))

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00006)

    # 数据集所在根目录
    # # http://download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="D:/Dataset/SemanticSegmentation/AIRS_512/")
    # parser.add_argument('--data-path', type=str,
    #                     default="D:/Dataset/SemanticSegmentation/AIRS/trainval/")
    parser.add_argument('--data-path', type=str,
                        default="D:/Dataset/SemanticSegmentation/2021LoveDA/")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='pretrained_weights/upernet_swin_small_patch4_window7_512x512.pth',
                        help='initial weights path')
    # parser.add_argument('--weights', type=str, default='',
    #                     help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save-path', type=str, default='loveda_urban_upernet_swin_small_')
    opt = parser.parse_args()

    main(opt)