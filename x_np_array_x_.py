
x=np.array(x)
y=np.array(y)
x.shape,y.shape
y=y[:,None]
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=0)
#归一化
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

y_train_onehot=tf.keras.utils.to_categorical(y_train)
y_test_onehot=tf.keras.utils.to_categorical(y_test)

