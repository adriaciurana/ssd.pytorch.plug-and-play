import cv2, glob, argparse, torch
from ssd import SSD
from dataset.logos.logo_dataset import LogoDataset
str2bool = lambda x: x.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(description='SSD plug & play test')
parser.add_argument('--weights', default='checkpoints/model_300_vgg16_final_logos.pth.tar',
                    type=str, help='Checkpoint of the model')
parser.add_argument('--cuda', default=True, type=str2bool, help='Enable or not cuda')
parser.add_argument('--test_filenames', default='test_images/*.jpg', type=str, help='Regex of filenames')
args = parser.parse_args()

net = SSD(cuda=args.cuda, architecture='300_VGG16', num_classes=len(LogoDataset.CLASSES))
has_cuda = args.cuda and torch.cuda.is_available()
if has_cuda:
    weights = torch.load(args.weights)['model']
else:
    weights = torch.load(args.weights, map_location='cpu')['model']
net = SSD.load(weights=weights)

COLORMAP = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
for filename in glob.glob(args.test_filenames):
    im = cv2.imread(filename)
    results = net.predict(im)
    for i, result in enumerate(results):
        print(LogoDataset.CLASSES[result['class']])
        class_ = LogoDataset.CLASSES[result['class']]
        position = result['position']
        confidence = int(100*result['confidence'])

        cv2.rectangle(im, (int(position[0]), int(position[1])), (int(position[2]), int(position[3])), COLORMAP[i % len(COLORMAP)])
        cv2.putText(im, '%s (%d%%)' % (class_, confidence), (int(position[0]), int(position[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORMAP[i % len(COLORMAP)], 2, cv2.LINE_AA)

    cv2.imshow('SSD detections & recognitions', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
