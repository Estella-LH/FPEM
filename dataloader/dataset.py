import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, training, images_dir_list, meta_file_list, score_key_list, resize_size, is_pair=False, pad_mode='rightdown'):
        super(FaceDataset, self).__init__()
        self.training = training
        self.resize_size = resize_size
        metas_list = [pd.read_csv(meta_file) for meta_file in meta_file_list]
        self.metas_combine = pd.concat(metas_list, ignore_index=True)
        self.images_dir_list = images_dir_list
        self.score_key_list = score_key_list
        self.to_tensor = transforms.ToTensor()
        #self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.flip = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip()])
        self.is_pair = is_pair
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.metas_combine) if not self.is_pair else len(self.metas_combine) // 2

    def __getitem__(self, idx):

        def get_by_idx(idx):
            image_path = self.get_image_path(idx)
            image = self.opencv_read_image(image_path)
            image = self.to_tensor(image)
            image_face = self.crop_image_with_bbox(image, self.metas_combine.loc[idx, 'bbox'])
            # if self.training:
            #     image_face = self.flip(image_face)

            # result_dict = {'image_path': image_path}
            result_dict = {}
            for i, size in enumerate(self.resize_size, start=1): # multiple resizes according to input
                resized_image = self.proport_resize_and_pad(image_face, size, self.pad_mode)
                variable_name = f'image_input_{i}'
                result_dict[variable_name] = resized_image        
            label = torch.tensor([self.metas_combine.loc[idx, score_key] for score_key in self.score_key_list], dtype=torch.float32)
            # ann_cnt = label[1]+label[2]+label[3]+label[4]+label[5]
            # p1, p2, p3, p4, p5 = label[1]/ann_cnt, label[2]/ann_cnt, label[3]/ann_cnt, label[4]/ann_cnt, label[5]/ann_cnt
            # result_dict['label'] = torch.tensor([p1, p2, p3, p4, p5])
            # print(result_dict['label'])
            result_dict['label'] = label 

            return result_dict
        
        if self.is_pair:
            return get_by_idx(idx * 2), get_by_idx(idx * 2 + 1)

        return get_by_idx(idx)
    
    def get_image_path(self, idx):
        image_name = self.metas_combine.loc[idx, 'img']
        image_path = ''
        for images_dir in self.images_dir_list:
            path = os.path.join(images_dir, image_name)
            if os.path.exists(path):
                image_path = path
                break
        if not image_path:
            raise FileNotFoundError('No such %s in Images directorys' % image_name)

        return image_path
    def opencv_read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def crop_image_with_bbox(self, image, bbox):
        x, y, w, h = list(map(int, bbox.split()))
        coor_offset = 2
        return image[:, y+coor_offset:y+h-coor_offset, x+coor_offset:x+w-coor_offset]

    def proport_resize_and_pad(self, image, target_size, pad_mode='rightdown'):
        _, h, w = image.size()
        scale_factor = target_size / max(h, w)
        scale_h = int(h * scale_factor)
        scale_w = int(w * scale_factor)

        if pad_mode == 'rightdown':
            pad_left, pad_top, pad_right, pad_bottom = 0, 0, target_size - scale_w, target_size - scale_h
        elif pad_mode == 'surround':
            pad_left, pad_top = (target_size - scale_w) // 2, (target_size - scale_h) // 2
            pad_right, pad_bottom = target_size - scale_w - pad_left, target_size - scale_h - pad_top
        else:
            raise ValueError('Unknown padding mode: %s' % pad_mode)

        transform = transforms.Compose([
            transforms.Resize((scale_h, scale_w)),
            transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')])

        return transform(image)

if __name__ == '__main__':
    # images_dir_list = [r'D:\Work\VQA_DATA\debug\dir1', r'D:\Work\VQA_DATA\debug\dir2']
    # meta_file_list =[r'D:\Work\VQA_DATA\debug\1.csv', r'D:\Work\VQA_DATA\debug\2.csv']
    # dataset = FaceDataset(True, images_dir_list, meta_file_list, \
    #         ['fuse', 'mopi', 'xiaci', 'bujun', 'meixing', 'meizhuang', 'yanzhi'], 224)
    # for i in range(5):
    #     print(dataset.__getitem__(i))

    images_dir_list = ['/mnt/nfs1/hongjiu/data/MEIYAN_train_data/meizhuang']
    meta_file_list =['/mnt/nfs1/hongjiu/data/meta_files/temp.csv']
    dataset = FaceDataset(True, images_dir_list, meta_file_list, \
            ['mos_clean'], [112,160], pair=True)
    for i in range(6):
        a, b = dataset.__getitem__(i)
        print(a['image_path'])
        print(b['image_path'])