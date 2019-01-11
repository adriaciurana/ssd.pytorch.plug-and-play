import argparse, torch, os, sys, time, math, shutil, threading
from ssd import SSD
from imgaug import augmenters as iaa
from dataset.logos.logo_dataset import LogoDataset

"""
	MAIN ARGUMENTS
"""
str2bool = lambda x: x.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(description='SSD plug & play training')

# Architecture settings
parser.add_argument('--architecture', default='300_VGG16', type=str, help='Based architecture to construct SSD', choices=list(SSD.ARCHITECTURES.keys()))
parser.add_argument('--pretrained_base', default=True, type=str2bool, help='Use pretrained based architecture')
parser.add_argument('--resume', default=None, type=str, help='Saved state_dict of the model')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda or not')
parser.add_argument('--dataset_name', default="logos", type=str, help='Dataset name')

# Learning settings
parser.add_argument('--init', default=None, type=str, help='Saved state_dict of the model')
parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay value')
parser.add_argument('--num_workers', default=4, type=int, help='Number of dataloader workers')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size in training')
parser.add_argument('--batch_size_val', default=20, type=int, help='Batch size in validation')
parser.add_argument('--start_epochs', default=1, type=int, help='Start epochs')
parser.add_argument('--epochs', default=200, type=int, help='Epochs')
parser.add_argument('--iterations', default=1000, type=int, help='Iterations in training')
parser.add_argument('--iterations_monitor', default=10, type=int, help='Iterations monitor in training')
parser.add_argument('--iterations_val', default=50, type=int, help='Iterations in validation')

# Visualization settings
parser.add_argument('--tensorboardX', default=True, type=str2bool, help='Use tensorboardX during learning')
parser.add_argument('--log_folder', default='log', type=str, help='Folder to save logs')

# Save models
parser.add_argument('--save_folder', default='checkpoints', type=str, help='Folder to save checkpoints')
parser.add_argument('--save_name', default='model_{architecture}_{epoch}_{dataset_name}.pth.tar', type=str, help='Name of model checkpoint')
parser.add_argument('--save_interval', default=1, type=int, help='Epochs interval to save')


args = parser.parse_args()

# Show summary
print('Summary:')
print('======================================')
for k in args.__dict__:
    data = args.__dict__[k]
    if k == 'save_name':
        data = data.format(architecture=args.architecture, epoch="{epoch}", dataset_name=args.dataset_name)
    print('%s: %s' % (k, data))
print('======================================')

"""
    CUDA
"""
has_cuda = args.cuda and torch.cuda.is_available()

"""
    CHECK HAVE TENSORBOARDX
"""
if args.tensorboardX:
    has_tensorboardX = True
    try:
        import tensorboardX
        log_folder = args.log_folder + "/{architecture}_{dataset_name}".format(architecture=args.architecture, dataset_name=args.dataset_name)
        train_writer = tensorboardX.SummaryWriter(log_folder + "/training")
        if args.iterations_val > 0:
            val_writer = tensorboardX.SummaryWriter(log_folder + "/validation")
    except:
        has_tensorboardX = False
else:
    has_tensorboardX = False

"""
    DATA AUGMENTATION
"""
AUGMENTERS = [
    # Flip Images left-right
    iaa.Fliplr(0.5),
    
    # Change Contrast
    iaa.Multiply((1, 1.5)),
    
    # Apply affine transforms with -20,20 pixels of translation, 0.5, 1.2 of scales, -30,30 of rotation and -10, 10 of shear.
    iaa.Affine(
        translate_px={"x": (-20, 20), "y": (-20, 20)},
        scale=(0.5, 1.2),
        rotate=(-30, 30),
        shear=(-10, 10)
    ),

    # With a 0.5 probability, we select only one of these elements
    iaa.Sometimes(0.5, 
        iaa.OneOf([
            # Apply a contrast normalization
            iaa.ContrastNormalization((0.75, 1.5)),

            # Add noise
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

            # Smooth image
            iaa.GaussianBlur(sigma=(0, 3.0))
        ])
    )
]

"""
    MODEL
"""
model_net = SSD(architecture=args.architecture, cuda=has_cuda, pretrained=args.pretrained_base, num_classes=len(LogoDataset.CLASSES))
train_writer.add_graph(model_net, torch.rand(1, 3, model_net.image_size, model_net.image_size))

if has_cuda:
    net = torch.nn.DataParallel(model_net)
    torch.backends.cudnn.benchmark = True
else:
    net = model_net

# Load weights
def init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

if args.resume is not None:
    print('Loading checkpoint / model...')
    model_net.load_model(torch.load(args.resume)['model'])

else:
    # Init weights of base
    if not args.pretrained_base:
        print('Init pretrained base...')
        model_net.base_net.apply(init)
    else:
        print('Loading pretrained base...')

    # Init the rest of weights
    print('Init misc, extras, locations and confidences...')
    model_net.apply_only_non_base(init)

"""
    DATASET DEFINITION
"""
train_dataset = LogoDataset(root=os.path.dirname(os.path.abspath(__file__)) + '/dataset/logos/data', transform=SSD.Utils.Transform(AUGMENTERS, model_net.image_size))
val_dataset = LogoDataset(root=os.path.dirname(os.path.abspath(__file__)) + '/dataset/logos/data', transform=SSD.Utils.Transform(None, model_net.image_size))
size_train_dataset = int(0.8 * len(train_dataset))
indices = torch.randperm(len(train_dataset)).cpu()
train_dataset, val_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[:size_train_dataset]), torch.utils.data.dataset.Subset(val_dataset, indices[size_train_dataset:])

"""
    OPTIMIZER DEFINITION
"""
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

"""
    START TRAINING
"""
data_loader_train = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=SSD.Utils.collate_fn, pin_memory=True)
data_loader_val = torch.utils.data.DataLoader(val_dataset, args.batch_size_val, num_workers=args.num_workers, shuffle=True, collate_fn=SSD.Utils.collate_fn, pin_memory=True)

# Main for to train
def train(epoch):
    net.train()

    batch_iter = iter(data_loader_train)
    loss_mean = 0
    loss_counter = 0

    loss_mean_monitoring = 0
    loss_counter_monitoring = 0
    time_mean_monitoring = 0
    time_counter_monitoring = 0
    for iteration in range(args.iterations):
        # Load data
        try:
            images, targets = next(batch_iter)
        except StopIteration:
            batch_iter = iter(data_loader_train)
            images, targets = next(batch_iter)

        if has_cuda:
            images = images.cuda()
            targets = [target.cuda() for target in targets]
        
        t0 = time.time()
        
        optimizer.zero_grad()
        
        loss, _ = model_net.compute_loss(images, targets, net=net)
        if not math.isinf(loss):
            loss_mean += loss.item()
            loss_counter += 1

            loss_mean_monitoring += loss.item()
            loss_counter_monitoring += 1
        loss.backward()
        
        optimizer.step()
         
        time_mean_monitoring += time.time() - t0
        time_counter_monitoring += 1


        if iteration > 0 and (iteration + 1) % args.iterations_monitor == 0:
            print('epoch: %d || iteration: %d || Loss: %.4f ||' % (epoch, iteration + 1, loss_mean_monitoring / loss_counter_monitoring), end=' ')
            print('timer: %.4f sec.' % (time_mean_monitoring / args.iterations_monitor))
                        
            loss_mean_monitoring = 0
            loss_counter_monitoring = 0
            time_mean_monitoring = 0
            time_counter_monitoring = 0
    if loss_counter == 0:
        return 0
    return loss_mean / loss_counter
        

def validation(epoch):
    net.eval()
    with torch.no_grad(): 
        batch_iter = iter(data_loader_val)
        loss_mean = 0
        loss_counter = 0
        t0 = time.time()
        for iteration in range(args.iterations_val):
            # Load data
            try:
                images, targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader_val)
                images, targets = next(batch_iter)

            if has_cuda:
                images = images.cuda()
                targets = [target.cuda() for target in targets]

            loss, _ = model_net.compute_loss(images, targets, net=net)
            if not math.isinf(loss):
                loss_mean += loss.item()
                loss_counter += 1
            
        t1 = time.time()
        print('epoch %d || validation || Loss: %.4f ||' % (epoch, loss_mean / args.iterations_val), end=' ')
        print('timer: %.4f sec.' % (t1 - t0))

    if loss_counter == 0:
        return 0
    return loss_mean / loss_counter
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs(args.save_folder, exist_ok=True)
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, best_name)

best_val = float('inf')
best_name = args.save_folder + "/" + args.save_name.format(architecture=args.architecture, epoch="best", dataset_name=args.dataset_name)
for epoch in range(args.start_epochs - 1, args.epochs):
    epoch_1 = epoch + 1
    loss_train = train(epoch_1)
    if args.iterations_val > 0:
        loss_val = validation(epoch_1)
    else:
        loss_val = loss_train

    if best_val > loss_val:
        is_best = True
        best_val = loss_val
    else:
        is_best = False

    if has_tensorboardX:
        threading.Thread(target= lambda: train_writer.add_scalar('Loss', loss_train, int(epoch_1))).start()
        if args.iterations_val > 0:
            threading.Thread(target= lambda: val_writer.add_scalar('Loss', loss_val, int(epoch_1))).start()

    if epoch_1 % args.save_interval == 0:
        save_checkpoint({
            'epoch': epoch_1,
            'model': model_net.save_model(),
            'best_loss': best_val,
        }, is_best, args.save_folder + "/" + args.save_name.format(architecture=args.architecture, epoch=epoch_1, dataset_name=args.dataset_name))
        

