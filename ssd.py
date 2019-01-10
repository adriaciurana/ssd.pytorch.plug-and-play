"""
Much of the code has been influenced by the following implementations:
- https://github.com/amdegroot/ssd.pytorch
- https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad

¡That is why all the gratitude belongs to them!

This implementation tries to simplify the learning process of an SSD in different ways:
- It is a single file + models folder, which allows a quick adaptation to the project where you want to use it.
- It is easy to use this scheme to implement other architectures such as SSD-512, Mobilenet, etc ... 
  Which allows a correct generalization of models based on the original SSD.
- It is extremely documented what the code tries to do understandable to all and thus be able to improve it continuously.
"""

import torch, torchvision, numpy as np, cv2, itertools, math, imgaug as ia
from architectures import ARCHITECTURES
class SSD(torch.nn.Module):
    """
        DEFINITIONS
    """
    ARCHITECTURES = ARCHITECTURES
    EPS = 1e-10
    class Priors:
        class Map(object):
            def __init__(self, image_size, priors, has_cuda):
                """
                Args:
                    image_size: size of images
                    priors: configuration of priors
                """
                self.image_size = image_size
                self.clip = priors['clip']
                self.configs = priors['configs'](SSD.Priors.Def)
                self.has_cuda = has_cuda
                self.generate_priors()

            def generate_priors(self):
                boxes_centered = []
                for prior in self.configs:
                    # Compute the scale to convert to [0, 1] coordinates.
                    scale = self.image_size / prior.step
                    
                    # Compute the size limits
                    s_k = prior.min_size/self.image_size
                    s_k_prime = math.sqrt(s_k * prior.max_size/self.image_size)

                    # Recorremos todo el espacio de features para generar cada prior
                    for j, i in itertools.product(range(prior.num_dim), repeat=2):
                        # Centro de la ventana
                        center_x = (i + 0.5) / scale
                        center_y = (j + 0.5) / scale

                        # Aspect ratio 1:1
                        # Small size
                        boxes_centered.append((center_x, center_y, s_k, s_k))
                        
                        # Big size
                        boxes_centered.append((center_x, center_y, s_k_prime, s_k_prime))
                        
                        # Aspect ratios x:1 or 1:x
                        for aspect_ratio in prior.aspect_ratios:
                            boxes_centered.append((center_x, center_y, s_k*math.sqrt(aspect_ratio), s_k/math.sqrt(aspect_ratio)))
                            boxes_centered.append((center_x, center_y, s_k/math.sqrt(aspect_ratio), s_k*math.sqrt(aspect_ratio)))
                # back to torch land
                priors = torch.Tensor(boxes_centered).view(-1, 4)
                if self.has_cuda:
                    priors = priors.cuda()
                
                if self.clip:
                    priors.clamp_(min=0.0, max=1.0)

                # Definitions
                # center, size in [0, 1] coordinates
                self.normal = priors
                
                # xmin, ymin, xmax, ymax in [0, 1] coordinates
                self.point_form = torch.cat((priors[:, :2] - priors[:, 2:] / 2, priors[:, :2]  + priors[:, 2:] / 2), dim=1)
                
                # xmin, ymin, width, height in [0, 1] coordinates
                self.bbox = self.point_form.clone()
                self.bbox[:, :2] -= self.bbox[:, 2:] 
                
                # only centers in [0, 1] coordinates
                self.centers = priors[:, :2]
                
                # only sizes in [0, 1] coordinates
                self.sizes = priors[:, 2:]

            def __len__(self):
                return self.bbox.size(0)

        class Def(object):
            def __init__(self, num_dim, step, size, aspect_ratios):
                """
                Args:
                    num_dim: num of dimensions of the feature map
                    step: step
                    size: max and min sizes
                    aspect_ratios: aspect ratios of this prior.
                """
                self.num_dim = num_dim
                self.step = step
                self.size = size
                self.aspect_ratios = aspect_ratios

            @property
            def num(self):
                return 2 + 2 * len(self.aspect_ratios)

            @property
            def min_size(self):
                return self.size[0]


            @property
            def max_size(self):
                return self.size[1]

    """
        INIT
    """
    def __init__(self, 
        num_classes,
        architecture='300_VGG16',
        cuda=True,
        pretrained=True, 
        thresholds={
            # Train
            'train_iou_overlap': 0.5,
            'train_alpha_loss': 1,
            'train_negative_positive_ratio': 3,
            
            # Prediction
            'prediction_confidence_threshold': 0.01,
            'prediction_top_k': 200,
            'prediction_iou_nms': 0.5,

            # Encode
            'encode_variances': [0.1, 0.2]
        }):
        """
        Constructor Single-file SSD that allow to create a 

        Args:
            num_classes: number of classes without count background
            architecture: Selected SSD architecture.
            cuda: enable or not cuda devices.
            pretrained: load or not a pretrained base model
            thresholds:
                - train_iou_overlap: minimum IoU to consider that prior and target location are overlapped.
                - train_alpha_loss: balance between confidence_loss and alpha*location_loss.
                - train_negative_positive_ratio: In order to fight the high imbalance data (num positives vs num negatives). Define a maximum ratio of negatives. For instance, 3:1.

                - prediction_confidence_threshold: Minimum confidence_threshold to consider in prediction stage (pre-filter of detection threshold).
                - prediction_top_k: Compare all boxes between them are too much expensive. In order to fligh these we select only the top_k bests for each class.
                - prediction_iou_nms: Minimum IoU to consider two priors are the same bbox (obviusly same class).
                
                - encode_variances: to relax center and size conditions in locations_loss.
        """

        super(SSD, self).__init__()
        self.has_cuda = torch.cuda.is_available() and cuda
        if self.has_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Define the architecture that are using
        self.architecture = SSD.ARCHITECTURES[architecture]

        # Define the number of classes
        self.num_classes = num_classes
        self.num_classes_plus_unknown = num_classes + 1 # Include the "background" class as "0"

        # Generate the priors
        self.image_size = self.architecture.image_size
        self.priors = SSD.Priors.Map(self.architecture.image_size, self.architecture.priors, self.has_cuda)

        # Define thresholds
        self.thresholds = thresholds
        
        """
            BASE 
        """
        # - base: Model base
        # - base_misc: Other added layers importants to execute correcly the base
        # - out_channels of last layer base
        # - out channels of layers with confidence and location
        self.base_net, base_misc, out_channels_base, self.out_channels_of_layers_with_classification = self.architecture.base(pretrained=pretrained)
        self.base_misc = torch.nn.Module()
        for k, v in base_misc.items():
            setattr(self.base_misc, k, v)

        """
            EXTRAS
        """
        # Add the extra layers and check which layer requiere confidences and locations. If this layer need classification we put in extras_with_classification
        self.extras = []
        self.extras_with_classification = []
        
        # Legend:
        # M -> MaxPool
        # C -> Convolution
        # Cc -> Convolution with classification
        in_channels = out_channels_base
        for type_layer, layer in self.architecture.extras:
            with_classification = False
            
            if type_layer[0] == 'M':
                # MaxPool
                layer_to_add = torch.nn.MaxPool2d(**layer)

            elif type_layer[0] == 'C':
                # Convolution
                layer_to_add = torch.nn.Conv2d(in_channels=in_channels, **layer)

                # With classification?
                if len(type_layer) == 2 and type_layer[1] == 'c':
                    # Add in the layers_with_classification
                    self.out_channels_of_layers_with_classification.append(layer_to_add.out_channels)
                    with_classification = True

            # Add in extras
            self.extras.append(layer_to_add)
            self.extras_with_classification.append(with_classification)

            # Update the in_channels only if has changed.
            if 'out_channels' in layer:
                in_channels = layer['out_channels']

        # Convert extras to ModuleList
        self.extras = torch.nn.ModuleList(self.extras)

        """
            CLASSIFICATION AND LOCATION
        """
        # Now add the locations and confidences layers
        self.locations = []
        self.confidences = []
        for i, prior in enumerate(self.priors.configs):
            # Get the number of priors by this layer
            num_priors = prior.num

            # Obtain the number of features of the layer with classification
            in_channels = self.out_channels_of_layers_with_classification[i]

            # Add the location num_priors * 4 -> num_priors x (x1, y1, x2, y2)
            self.locations.append(torch.nn.Conv2d(in_channels, num_priors * 4, kernel_size=3, padding=1))

            # Add the location num_priors * num_classes + 1 -> num_priors x (c0, ... cn+1), where c0 is background
            self.confidences.append(torch.nn.Conv2d(in_channels, num_priors * self.num_classes_plus_unknown, kernel_size=3, padding=1))

        # Convert locations and confidences to ModuleList
        self.locations = torch.nn.ModuleList(self.locations)
        self.confidences = torch.nn.ModuleList(self.confidences)

    """
        FORWARD
    """
    def forward(self, x):
        """
        Args:
            x: batch of images.
        """
        # Apply the base model
        """if self.has_cuda:
            x = x.cuda()"""
        x, results_with_classification = self.architecture.forward(self.base_net, x, self.base_misc._modules)
        
        # Apply extras
        for extra, with_classification in zip(self.extras, self.extras_with_classification):
            x = torch.nn.functional.relu(extra(x), inplace=True)
            if with_classification:
                results_with_classification.append(x)

        # Feed locations and confidence losses
        locations_results = []
        confidences_results = []
        for i, (loc, conf) in enumerate(zip(self.locations, self.confidences)):
            x = results_with_classification[i]
            partial_loc = loc(x).permute(0, 2, 3, 1).contiguous()
            partial_conf = conf(x).permute(0, 2, 3, 1).contiguous()
            locations_results.append(partial_loc.view(partial_loc.size(0), -1, 4))
            confidences_results.append(partial_conf.view(partial_conf.size(0), -1, self.num_classes_plus_unknown))

        # Refactorize locatiosn and confidences
        locations_results = torch.cat(locations_results, dim=1)
        confidences_results = torch.cat(confidences_results, dim=1)

        return locations_results, confidences_results

    """
        TRAINING
    """
    @staticmethod
    def IoU(box_a, box_b):
        """
        Args:
            box_a: box in x1, y1, x2, y2
            box_b: box in x1, y1, x2, y2
        """
        # Define sizes that we work
        size_box_a = box_a.size(0)
        size_box_b = box_b.size(0)

        def compute_area(box):
            """
            Args:
                box: box in x1, y1, x2, y2
            """
            sides = box[..., 2:] - box[..., :2]
            return sides[..., 0] * sides[..., 1]
        
        def compute_intersection(box_a, box_b):
            """
            Args:
                box_a: box in x1, y1, x2, y2
                box_b: box in x1, y1, x2, y2
            """
            
            # Define max and min of boxes
            box_a_min = box_a[:, :2]
            box_a_max = box_a[:, 2:]

            box_b_min = box_b[:, :2]
            box_b_max = box_b[:, 2:]

            """ 
            We apply the all against all by measuring: 
            minimum in the case of the maximum level to see which zone intersects.
            maximum in the case of the minimum level to see which area intersects.
            """
            max_intersected_xy = torch.min(box_a_max.unsqueeze(1).expand(size_box_a, size_box_b, 2), box_b_max.unsqueeze(0).expand(size_box_a, size_box_b, 2))
            min_intersected_xy = torch.max(box_a_min.unsqueeze(1).expand(size_box_a, size_box_b, 2), box_b_min.unsqueeze(0).expand(size_box_a, size_box_b, 2))
            sides = torch.clamp(max_intersected_xy - min_intersected_xy, min=0)
            
            # We calculate the interaction from the sides
            return sides[..., 0] * sides[..., 1] # size_box_a, size_box_b, 2

        intersection = compute_intersection(box_a, box_b)
        area_a = compute_area(box_a).unsqueeze(1).expand(size_box_a, size_box_b)
        area_b = compute_area(box_b).unsqueeze(0).expand(size_box_a, size_box_b)
        union = area_a + area_b - intersection # The common part in a and b is added twice, it is eliminated by the intection
        return intersection / union # A u B / A n B

    def encode(self, points):
        # points are center and size of [0, 1] coordinates
        center = (points[:, :2] + points[:, 2:]) / 2 - self.priors.centers
        center /= (self.thresholds['encode_variances'][0] * self.priors.sizes)

        size = (points[:, 2:] - points[:, :2]) / self.priors.sizes
        size = torch.log(size) / self.thresholds['encode_variances'][1] # se hace un logaritmo para no penalizar tanto y que la penalizacion tenga un crecimiento menor cuando esta muy alejada
            
        # Concatenamos las losses
        return torch.cat((center, size), dim=1)

    def decode(self, points):  
        decoded_boxes = torch.cat((
                self.priors.centers + points[:, :2] * self.thresholds['encode_variances'][0] * self.priors.sizes, 
                self.priors.sizes * torch.exp(points[:, 2:] * self.thresholds['encode_variances'][1])
        ), dim=1)
        decoded_boxes[:, :2] -= decoded_boxes[:, 2:] / 2
        decoded_boxes[:, 2:] += decoded_boxes[:, :2]
        return decoded_boxes

    def targets2priors(self, targets):
        """
        Args:
            targets: list of numpy of x1, y1, x2, y2, class in 0-1 coordinates
        """
        # First we calculate the IoU of the priors with the ground-truth
        batch_size = len(targets)
        locations_priors_encoded = torch.Tensor(batch_size, len(self.priors), 4)
        confidences_priors_labels = torch.LongTensor(batch_size, len(self.priors))
        if self.has_cuda:
            locations_priors_encoded = locations_priors_encoded.cuda()
            confidences_priors_labels = confidences_priors_labels.cuda()

        for batch_index, target in enumerate(targets):
            confidence_labels_batch = target[..., -1]
            locations_true_batch = target[..., :-1]
            """if self.has_cuda:
                confidence_labels_batch = confidence_labels_batch.cuda()
                locations_true_batch = locations_true_batch.cuda()"""
            
            # Compute IoUs
            ious = SSD.IoU(locations_true_batch, self.priors.point_form) # true x priors
            
            # Matching
            best_prior_overlap, best_prior_idx = ious.max(dim=1) # indexes_priors (x trues)
            best_truth_overlap, best_truth_idx = ious.max(dim=0) # indexes_trues (x priors)
            best_truth_overlap.index_fill_(0, best_prior_idx, 2) # make sure that all targets are selected at least once (if they do not coincide exactly with another target)
            best_truth_idx[best_prior_idx] = torch.arange(best_prior_idx.size(0))
            
            matched_priors_mask = best_truth_overlap >= self.thresholds['train_iou_overlap']
            matches = locations_true_batch[best_truth_idx] # num_priors x (x1 y1 x2 y2)

            confidences_priors_labels_single = confidence_labels_batch[best_truth_idx] + 1 # We convert the probability map for each bbox to one in priors. We skipped the class background (= 0)
            confidences_priors_labels_single[~matched_priors_mask] = 0 # All those that do not exceed a minimum iou are assigned as background (= 0)
            
            # Codification of center (mu-sigma normalization)
            locations_priors_encoded[batch_index] = self.encode(matches)
            
            # We concatenate the losses
            confidences_priors_labels[batch_index] = confidences_priors_labels_single

        """if self.has_cuda:
            locations_priors_encoded = locations_priors_encoded.cuda()
            confidences_priors_labels = confidences_priors_labels.cuda()"""
            
        return locations_priors_encoded, confidences_priors_labels

    def get_loss_predictions(self, predictions, targets):
        """
        Args:
            predictions: tuple of confidences and locations.
            targets: list of numpy of x1, y1, x2, y2, class in 0-1 coordinates
        """
        locations_pred, confidences_pred = predictions

        # Batch_size
        batch_size = len(targets)

        # Convert targets to priors
        locations_priors_encoded, confidences_priors_labels = self.targets2priors(targets)
        
        # Loss generation
        only_positives = confidences_priors_labels > 0
       
        # 1 - Localization loss
        locations_loss = torch.nn.functional.smooth_l1_loss(locations_pred[only_positives], locations_priors_encoded[only_positives], reduction='sum')
        
        # Global loss without reduction only to compute all values
        confidences_loss = torch.nn.functional.cross_entropy(confidences_pred.view(-1, self.num_classes_plus_unknown), confidences_priors_labels.view(-1), reduction='none').view(batch_size, -1)

        # 2a - Positive confidence loss
        confidences_positive_loss = confidences_loss[only_positives].sum()
        
        # 2b - Hard Negative Mining (Negative confidence loss)
        confidences_negative_loss = confidences_loss
        confidences_negative_loss[only_positives] = 0
        
        negative_selector = torch.clamp(self.thresholds['train_negative_positive_ratio'] * only_positives.sum(dim=1), max=only_positives.size(1) - 1)
        negative_selector = torch.cat(
                                [torch.cat((torch.ones(1*(negative_selector[i] > 0), int(negative_selector[i])), 
                                            torch.zeros(1*((only_positives.size(1) - negative_selector[i]) > 0), int(only_positives.size(1) - negative_selector[i]))), dim=1) 
                                 for i in range(negative_selector.size(0))], 
                            dim=0)
        confidences_negative_loss, _ = confidences_negative_loss.sort(dim=1, descending=True)
        confidences_negative_loss = (negative_selector * confidences_negative_loss).sum() #/ negative_selector.sum(dim=1)
        
        # 2c - Sum of both
        confidences_loss = confidences_positive_loss + confidences_negative_loss
        
        # 3 - Global loss
        loss = confidences_loss + self.thresholds['train_alpha_loss'] * locations_loss
        loss /= (only_positives.sum().float() + SSD.EPS) #.cuda().float()
        return loss

    def compute_loss(self, images, targets, normalize=True, net=None):
        """
        Args:
            images: Tensor of images
            targets: list of list of numpy of x1, y1, x2, y2, class in 0-1 coordinates
            net: allow to use data-parallel model
        """
        if normalize:
            images = self.normalize(images)
        
        if net is not None:
            predictions = net(images)
        else: 
            predictions = self(images)
        
        return self.get_loss_predictions(predictions, targets), predictions

    """
        PREDICTION
    """
    @staticmethod
    def box_nms(boxes, scores, threshold=0.5, mode='union'):
        '''Non maximum suppression.
        Args:
          boxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.
        Returns:
          keep: (tensor) selected indices.
        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        areas = (x2 - x1) * (y2 - y1)
        order = torch.arange(scores.size(0))
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=float(x1[i]))
            yy1 = y1[order[1:]].clamp(min=float(y1[i]))
            xx2 = x2[order[1:]].clamp(max=float(x2[i]))
            yy2 = y2[order[1:]].clamp(max=float(y2[i]))

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= threshold).nonzero().squeeze()
            
            if ids.numel() == 0:
                break
            
            order = order[ids + 1]
        return torch.LongTensor(keep)

    def detect(self, predictions, mode='union'):
        """
        Args:
            predictions: tuple of confidences and locations.
            mode: type of nms union
        """
        with torch.no_grad():
            locations_pred, confidences_pred = predictions
            confidences_pred = torch.nn.functional.softmax(confidences_pred, dim=-1)
            confidences_pred = confidences_pred.permute(0, 2, 1)

            # Batch size
            batch_size = locations_pred.size(0)

            # Classes
            #prior_classes = torch.arange(self.num_classes_plus_unknown).unsqueeze(0).expand_as(len(self.priors), -1)
            
            output = torch.zeros(batch_size, self.num_classes, self.thresholds['prediction_top_k'], 5)   
            for batch_index in range(batch_size):
                locations_decoded = self.decode(locations_pred[batch_index])

                """# Seleccionamos los top_k
                confidences_pred_sorted, confidences_pred_indexes = confidences_pred.sort(dim=0, descending=True)
                confidences_pred_sorted = confidences_pred_sorted[:self.confidence_top_k]
                confidences_pred_indexes = confidences_pred_indexes[:self.confidence_top_k]

                # Filtramos los resultados inferiores a un threshold
                only_positives = confidences_pred_sorted > self.confidence_threshold
                confidences_pred_sorted = confidences_pred_sorted[only_positives]
                confidences_pred_indexes = confidences_pred_indexes[only_positives]

                # Obtenemos las confidences, locatiosn y la clase a que pertenece
                positive_confidences = confidences_pred_sorted
                positive_locations = locations_decoded[confidences_pred_indexes].view(-1, 4)
                positive_classes = prior_classes[confidences_pred_indexes]"""



                #only_positives = confidences_pred[batch_index] > self.confidence_threshold
                #positive_confidences = confidences_pred[only_positives]
                #positive_locations = locations_decoded[only_positives].view(-1, 4)
                #positive_classes = prior_classes[only_positives]
                confidences_pred_per_batch = confidences_pred[batch_index]

                for class_ in range(1, self.num_classes_plus_unknown):
                    # Filtramos los resultados inferiores a un threshold
                    only_positives = confidences_pred_per_batch[class_] > self.thresholds['prediction_confidence_threshold']
                    if only_positives.sum() == 0:
                        continue

                    
                    positive_confidences = confidences_pred_per_batch[class_][only_positives]
                    positive_locations = locations_decoded[only_positives.unsqueeze(1).expand_as(locations_decoded)].view(-1, 4)
                    
                    confidences_pred_sorted, confidences_pred_indexes = positive_confidences.sort(dim=0, descending=True)
                    confidences_pred_sorted = confidences_pred_sorted[:self.thresholds['prediction_top_k']]
                    confidences_pred_indexes = confidences_pred_indexes[:self.thresholds['prediction_top_k']]
                    
                    positive_confidences = confidences_pred_sorted
                    positive_locations = positive_locations[confidences_pred_indexes, :]

                    selected_ids = SSD.box_nms(positive_locations, positive_confidences, threshold=self.thresholds['prediction_iou_nms'], mode=mode)
                    output[batch_index, class_ - 1, :len(selected_ids)] = torch.cat((positive_confidences[selected_ids].unsqueeze(1), positive_locations[selected_ids]), dim=1)
            
        return output
    
    def predict(self, images, normalize=True, threshold=0.5, mode='union', image_coordinates=True, is_BGR=True):
        """
        Args:
            images: tuple of images or image.
            normalize: apply normalization in image.
            threshold: minimum confidence to accept.
            mode: type of nms union.
            image_coordinates: return the output in image coordinates or 0-1 coordinates.
        """
        if not isinstance(images, (tuple, list)):
            images = [images]
        
        if image_coordinates:
            sizes = []
        
        new_images = np.empty(shape=(len(images), 3, self.image_size, self.image_size))
        for i, image in enumerate(images):
            if image_coordinates:
                sizes.append(image.shape[:2])
                image = cv2.resize(image, (self.image_size, self.image_size))
                if is_BGR:
                    image = image[..., ::-1]
            new_images[i] = image.transpose(2, 0, 1)

        images = torch.from_numpy(new_images).float()
        
        if normalize:
            images = self.normalize(images)

        if self.has_cuda:
            images = images.cuda()

        predictions = self(images)
        detections = self.detect(predictions, mode=mode)

        outputs = []
        batch_size, num_classes, num_top_k, _ = detections.size()
        for batch_index, size in enumerate(sizes):
            output_element = []
            for class_ in range(num_classes):
                for i in range(num_top_k):
                    if detections[batch_index, class_, i, 0] >= threshold:
                        confidence = detections[batch_index, class_, i, 0].cpu().detach().numpy()
                        position = detections[batch_index, class_, i, 1:].clamp(min=0.0, max=1.0)

                        if image_coordinates:
                            position *= torch.Tensor([size[1], size[0], size[1], size[0]])
                        
                        output_element.append({'class': class_, 'confidence': float(confidence), 'position': position.cpu().detach().numpy()})
                    else:
                        break
            outputs.append(output_element)
        return outputs

    """
        UTILS
    """
    def apply_only_non_base(self, x):
        self.base_misc.apply(x)
        self.extras.apply(x)
        self.locations.apply(x)
        self.confidences.apply(x)

    def normalize(self, x):
        """
        Args:
            x: Tensor batch of images.
        """
        x = x.float() / 255.0 # ToTensor
        
        # Apply normalization
        x[:, 0, :, :] -= self.architecture.normalization.mean[0]
        x[:, 1, :, :] -= self.architecture.normalization.mean[1]
        x[:, 2, :, :] -= self.architecture.normalization.mean[2]
        x[:, 0, :, :] /= self.architecture.normalization.std[0]
        x[:, 1, :, :] /= self.architecture.normalization.std[1]
        x[:, 2, :, :] /= self.architecture.normalization.std[2]

        return x

    def load_model(self, weights):
        self.load_state_dict(weights['state_dict'])

    @staticmethod
    def load(weights, cuda=True):
        model = SSD(cuda=cuda, architecture=weights['architecture'], num_classes=weights['num_classes'])
        model.load_model(weights)
        return model

    def save_model(self):
        return {
            'state_dict': self.state_dict(),
            'architecture': self.architecture.NAME,
            'num_classes': self.num_classes,
        }

    class Utils(object):
        @staticmethod
        def collate_fn(batch):
            """
                Args:
                    batch: list of tuples (image, target)
            """
            images = []
            targets = []
            for image, target in batch:
                images.append(image)
                targets.append(torch.FloatTensor(target))
            return torch.stack(images, dim=0), targets
        
        class Transform(object):
            def __init__(self, augmenters, image_size, is_BGR=True):
                """
                Args:
                    augmenters: list of imgaug library augmenters to use.
                    is_BGR: specify the order of channels in images. OpenCV use BGR
                """
                self.image_size = image_size
                self.is_BGR = is_BGR
                if augmenters is not None and len(augmenters) > 0:
                    self.seq = ia.augmenters.Sequential(augmenters)
                else:
                    self.seq = None

            def apply_augmenters(self, image, target):
                """
                Args:
                    image: image to apply augmentation
                    target: target to apply augmentation
                """
                if self.seq is None:
                    target = np.array(target, dtype=np.float32)
                    target[:, 0] /= image.shape[1]
                    target[:, 1] /= image.shape[0]
                    target[:, 2] /= image.shape[1]
                    target[:, 3] /= image.shape[0]
                    
                else:
                    seq_det = self.seq.to_deterministic()

                    image = seq_det.augment_images([image])[0]
                    
                    boxes = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=target[i][0], y1=target[i][1], x2=target[i][2], y2=target[i][3]) for i in range(target.shape[0])], shape=image.shape)
                    boxes = seq_det.augment_bounding_boxes([boxes])[0]


                    image_after = boxes.draw_on_image(image, thickness=2, color=[0, 0, 255])
                    target = np.array([[box_aug.x1 / image.shape[1], 
                                       box_aug.y1 / image.shape[0], 
                                       box_aug.x2 / image.shape[1], 
                                       box_aug.y2 / image.shape[0],
                                       target[i][-1]] for i, box_aug in enumerate(boxes.bounding_boxes)], dtype=np.float32)

                if self.is_BGR:
                        image = image[..., ::-1]
                
                target[:, :4] = target[:, :4].clip(min=0.0, max=1.0)
                image = cv2.resize(image, (self.image_size, self.image_size))
                return image, target

            def __call__(self, image, target):
                # Apply augmenters, resize and transform coordinates to 0-1
                image, target = self.apply_augmenters(image, target)

                return image, target


                
