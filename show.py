
la=[str(i) for i in range(1,9)]
def show(a,b,c,d):
    
    
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].set_title('accuracy of train and valuation')
    axes[0].plot(la,train_history.history[a],marker='*')
    axes[0].plot(train_history.history[b],marker='*')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    
    aa=round(train_history.history[a][7],2)
    bb=round(train_history.history[b][7],2)
    
    axes[0].text(la[7],train_history.history[a][7],aa,ha='center',va='bottom')
    axes[0].text(la[7],train_history.history[b][7],bb,ha='center',va='top')
    #axes[0].set_xticks(la,['as','asd',3,4,5,6,7,8])
#     for x1,y1 in zip(la,train_history.history[a]):
#         y1=round(y1,2)
#         axes[0].text(x1,y1,y1,ha='center',va='bottom',fontsize=10,c='b')
        
    axes[0].legend(['train_accuracy','val_accuracy'])
    
    axes[1].set_title('loss of train and valuation')
    axes[1].plot(la,train_history.history[c],marker='*')
    axes[1].plot(train_history.history[d],marker='*')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    
    cc=round(train_history.history[c][7],2)
    dd=round(train_history.history[d][7],2)
    
    axes[1].text(la[7],train_history.history[c][7],cc,ha='center',va='top')
    axes[1].text(la[7],train_history.history[d][7],dd,ha='center',va='bottom')
    axes[1].legend(['train_loss', 'val_loss'])
    #axes[1].show()

show('accuracy','val_accuracy','loss','val_loss')
