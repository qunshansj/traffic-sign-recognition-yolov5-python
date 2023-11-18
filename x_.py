
x=[]
y=[]
with open ('./target.txt','r') as f:
    for j,i in enumerate(f):
        path=i.split()[0]
        lable=i.split()[1]
        print('读取第%d个图片'%j,path,lable)
        src=cv2.imread('./suju/'+path)
        x.append(src)
        y.append(int(lable))

