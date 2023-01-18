import os
import torch
import torch.utils.data as data

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # output folder
        self.outdir = 'test_result'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                               transform=input_transform)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        # create network
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=True, root=args.save_folder).to(args.device)
        print('Finished loading model!')

        self.metric = SegmentationMetric(2)


    def eval(self):
        l = len(self.val_loader)
        self.model.eval()
        for i, (image, label) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            print(f"Val-Example {i} out of {l}")
            outputs = self.model(image)

            # we have to cheat a little, and ignore the edgecase where rider and person are close by
            t1 = outputs[0].squeeze()
            confs = t1[11] + t1[12]
            
            first = torch.argmax(outputs[0], 1)
            pred = first.cpu().data.numpy()
            pred = (pred == 11) | (pred == 12)
            label = (label == 11) | (label == 12)
            label = label.numpy()

            self.metric.update(loss = 0, preds = pred, labels = label, elapsed_time = 1, confidences = confs)
            res_dict = self.metric.get()
            for k,v in zip(res_dict.keys(), res_dict.values()):
                print(f"%s : %f" % (k,v))

            predict = pred.squeeze(0)
            #mask = get_color_pallete(predict, self.args.dataset)
            #mask.save(os.path.join(self.outdir, 'seg_{}.png'.format(i)))

            """
            accuracy : 0.992991
            IoU : 0.599668
            time : 1.000000
            loss : 0.000000
            auprc : 0.491734
            dice : 0.570811
            recall : 0.506056
            precision : 0.520647
            error : 0.007009
            """



if __name__ == '__main__':
    args = parse_args()
    args.device = 'cpu'
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    evaluator.eval()

