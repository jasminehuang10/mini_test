#用户：xiangjianqun   

#日期：2019-02-22   

#时间：22:41   

#文件名称：PyCharm


import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
            f.close()
    return np.mat(data),m,n

imgData,row,col = loadData('music.jpg')
label = KMeans(n_clusters=4).fit_predict(imgData)
label = label.reshape([row,col])
print(label)
#创建一张新的灰度保存聚类后的结果
pic_new = image.new("L", (row, col))
#根据所属类别像图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
pic_new.save("result-music-4.jpg", "JPEG")

# 设置指定位置的颜色
#img.putpixel((x, y),(color1....))
# 三个参数分别代表图像的模式：常用的为RGB(3通道) 、RGBA(4通道为透明通道，0为完全透明， 256为不透明)
# 第二个参数为图像的长宽参数
# 第三个为默认的填充颜色，RGB时长度为3，RGBA是长度为4
#img =Image.new('mod',(width,height),(color1, color2, color3, color4))
