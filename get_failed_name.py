# -*- coding: utf-8 -*-
from PIL import Image
from collections import OrderedDict
import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from model import build_model

class ImageClassificationService(object):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        # self.model = models.__dict__['resnet50'](num_classes=54)
        self.model = build_model(model_name, num_classes=54, pretrained=False)  # 生成模型， 自己修改模型
        self.use_cuda = False
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = OrderedDict()
            # 训练脚本 main.py 中保存了'epoch', 'arch', 'state_dict', 'best_acc1', 'optimizer'五个key值，
            # 其中'state_dict'对应的value才是模型的参数。
            # 训练脚本 main.py 中创建模型时用了torch.nn.DataParallel，因此模型保存时的dict都会有‘module.’的前缀，
            # 下面 tmp = key[7:] 这行代码的作用就是去掉‘module.’前缀
            for key, value in checkpoint['state_dict'].items():
                if key[:7] == 'module.':
                    tmp = key[7:]
                else:
                    tmp = key
                state_dict[tmp] = value
            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.idx_to_class = checkpoint['idx_to_class']
        self.normalize = transforms.Normalize(mean=[0.582, 0.523, 0.462],
                                         std=[0.298, 0.304, 0.325])

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])

        self.label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.transforms(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        img = data["input_img"]
        img = img.unsqueeze(0)

        if self.use_cuda:
            img = img.cuda()

        with torch.no_grad():
            pred_score = self.model(img)
            pred_score = F.softmax(pred_score.data, dim=1)
            if pred_score is not None:
                pred_label = torch.argsort(pred_score[0], descending=True)[:1][0].item()
                pred_label = self.idx_to_class[int(pred_label)]
                # result = {'result': self.label_id_name_dict[str(pred_label)]}
                result = self.label_id_name_dict[str(pred_label)]
            else:
                # result = {'result': 'predict score is None'}
                result = None

        return result, int(pred_label)

    def _postprocess(self, data):
        return data

    def inference(self, data):
        """
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        """
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()


        data, pred_label = self._inference(data)

        return data, pred_label


if __name__ == "__main__":
    model_name = 'se_resnext50'
    model_path = 'weights/se_resnext50/epoch_15_93.0.pth'   # 训练的权重
    service = ImageClassificationService(model_name, model_path)
    data_root = '../train_val500/val/'                      # 测试样本的根目录
    dest_root = osp.join(os.getcwd(),'hard_classify')       # 分错样本的存放目录

    classes = os.listdir(data_root)
    classes.sort()
    count = 0
    for cls in classes:
        sub_dir = osp.join(data_root, cls)
        file_names = os.listdir(sub_dir)
        file_names.sort()
        for file_name in file_names:
            full_path = os.path.join(sub_dir, file_name)
            data_dict = {"input_img": {file_name: full_path}}
            cls_name, id = service.inference(data_dict)
            if id != int(cls):
                count += 1
                print(full_path)
                shutil.copy(full_path, os.path.join(dest_root, "{}_{}_{}.jpg".format(cls, id, file_name)))

    print("分错的样本一共{}个....".format(count))
    print("样本保存在：----->{}<-----".format(dest_root))
    print("文件夹下面的文件命名格式：{真实类别}_{预测的错误类别}_{原始图像名称}.jpg\n"
          "--------------------------------------------------------------\n"
          "老弟，加油啊，给我介绍妹纸认识啊^V^.....  \n"
          "零花钱能不能到手就要看你的啦！！！！！     \n"
          "   ----▂▄▄▓▄▄▂----                   \n"
          "    ◢◤ █   ████▄▄▄▄◢◤                \n"
          "   █ 白嫖使你快乐 █…………………………╬         \n"
          "     ◥  ██████  ◤                    \n"
          "      ══╩══╩═══                      \n"
          "--------------------------------------------------------------\n"
          "老弟，加油啊，给我介绍妹纸认识啊^V^....."
          )