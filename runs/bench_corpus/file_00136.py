#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/6 9:35
# @File    : Compare.py
# @Software: PyCharm


import cv2
import numpy
from PIL import Image


class GraphicalLocator:

    def __init__(self, img_path, threshold_1=0.9, threshold_2=0.9):
        """
        @param img_path: 预期的图片路径
        @param threshold_1: 对比阀值参数一,可根据情况修改默认0.9
        @param threshold_2: 对比阀值参数二，可根据情况修改默认0.9
        """
        # 预期图片
        self.locator = img_path
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        # x、 y位置，从左上角开始计数，以像素为单位
        self.x = None
        self.y = None
        self.img = cv2.imread(img_path)
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.threshold = None
        self.result = True

    def find_and_check(self, drv):
        """
        @param drv: 实际截图的路径
        @return: 判断结果，True or Flase
        """
        # 坐标复位
        self.x = self.y = None
        # 打开图片
        scr = Image.open(drv)
        # 转换为OpenCV接受的格式
        scr = numpy.asarray(scr, dtype=numpy.float32).astype(numpy.uint8)
        # 将图像从BGR转换为RGB格式
        scr = cv2.cvtColor(scr, cv2.COLOR_BGR2RGB)
        # 通过matchTemplate方法来模板匹配图片，在通过minMax方法来得到匹配的矩阵数列,如果不能匹配，
        # 针对完全匹配不到的场景，直接返回Flase,成功匹配则进入下一步继续匹配
        try:
            img_match = cv2.minMaxLoc(
                cv2.matchTemplate(scr,
                                  self.img,
                                  cv2.TM_CCOEFF_NORMED))
        except Exception as e:
            self.result = False
            return self.result

        # 计算找到的元素的位置
        self.x = img_match[3][0]
        self.y = img_match[3][1]

        # 从完全截图裁剪部分匹配模板图像
        scr_crop = scr[self.y:(self.y + self.height),
                   self.x:(self.x + self.width)]

        # 计算两个模板的颜色直方图,并匹配图片进行比较
        scr_hist = cv2.calcHist([scr_crop], [0, 1, 2], None,
                                [8, 8, 8], [0, 256, 0, 256, 0, 256])
        img_hist = cv2.calcHist([self.img], [0, 1, 2], None,
                                [8, 8, 8], [0, 256, 0, 256, 0, 256])
        comp_hist = cv2.compareHist(img_hist, scr_hist,
                                    cv2.HISTCMP_CORREL)
        # 保存阈值匹配：图形图像和图像直方图
        self.threshold = {'shape': round(img_match[1], 2),
                          'histogram': round(comp_hist, 2)}
        print(self.threshold)
        # 返回对比判断结果,布尔值
        self.result = True if self.threshold['shape'] >= self.threshold_1 and self.threshold[
            'histogram'] >= self.threshold_2 else False
        return self.result


if __name__ == '__main__':
        result_1 = GraphicalLocator('F:\\ZHAF\\Locpic\\byzn\\6.png').find_and_check('F:\\ZHAF\\ScreenShots\\2020-11-16_screenshots_dir\\src-2020-11-16_16_40_35.png')
        print(result_1)