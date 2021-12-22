import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
from torch.nn import init
import gc
import numpy as np
import pickle
from pathlib import Path
import inspect




class Hparams:

    num_epochs: int = 50
    batch_size: int = 6
    learning_rate: int = 1e-3
    lr_decay_step_size: int = 50
    lr_decay_rate: int = 1
    weight_decay_rate: int = 0
    early_stop_patience: int = 50
    num_classes: int = 2



    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


    def device(self):
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        return torch.device('cpu')


    #make parameter class iterable
    def __iter__(self):
        def f(obj):
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("__") and not inspect.isfunction(v)
            }
        #vars()返回类属性的__dict__, 而实例化self后， __class__仍然存在实例化前的dict（全局变量），而self本身的__dict__属性值包括__init__时传入对象
        return iter({**f(self.__class__), **f(self)}.items())







class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.val_accu = None

    def __call__(self, val_loss, val_accu, model, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_accu, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_accu, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_accu, model, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, save_path)	# save the best model
        self.val_loss_min = val_loss
        self.val_accu = val_accu

# pytorch has bug with apply() function so we do not use manul initilizatioin this time
# #kaiming uniform initializer
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1:
#         init.kaiming_uniform_(m.weight.data)
#         init.constant_(m.bias.data, 0.0)
#     elif classname.find('Linear') != -1:
#         init.kaiming_uniform_(m.weight.data)
#         init.constant_(m.bias.data, 0.0)
#     elif classname.find('GRU') != -1:
#         init.kaiming_uniform_(m.weight_ih_l[0].data)
#         init.kaiming_uniform_(m.weight_hh_l[0].data)
#         init.kaiming_uniform_(m.bias_ih_l[0].data)
#         init.kaiming_uniform_(m.bias_hh_l[0].data)
#         init.kaiming_uniform_(m.weight_ih_l[1].data)
#         init.kaiming_uniform_(m.weight_hh_l[1].data)
#         init.kaiming_uniform_(m.bias_ih_l[1].data)
#         init.kaiming_uniform_(m.bias_hh_l[1].data)






class DL_process:
    '''Train, test, and predict the neural network'''

    def __init__(self, cfg, seed):

        # Set the random seed manually for reproducibility.
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.cfg = cfg


    def change_device(self, model):
        # print(self.cfg.device())
        return model.to(self.cfg.device())
        # #applied when more than one gpu applied
        # return torch.nn.DataParallel(self.model).to(cfg.device())


    # 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
    def get_variable(self, x):
        x = Variable(x)
        return x.to(self.cfg.device())


    # 绘制混淆矩阵 draw confusion matrix ( 对整个数组生效 )
    @staticmethod
    def confusion_matrix(preds, labels, conf_matrix):
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix


    #crnn training function
    def train_nn(self, nn_model, train_data_label, val_data_label, save_path):
        '''This function is to train the neural network with training set and validation set'''

        # # 初始化神经网络
        # pytorch has bug with apply() function so we do not use manul initilizatioin this time
        # nn_model.apply(weights_init)

        train_len = train_data_label.segment_len
        val_len = val_data_label.segment_len

        num_epochs = self.cfg.num_epochs
        batch_size = self.cfg.batch_size
        lr = self.cfg.learning_rate
        decay_step_size = self.cfg.lr_decay_step_size
        decay_rate = self.cfg.lr_decay_rate
        weight_decay_rate = self.cfg.weight_decay_rate
        estop_patience = self.cfg.early_stop_patience
        num_classes = self.cfg.num_classes

        #initialise the network model parameter
        nn_model.__init__()
        nn_model = self.change_device(nn_model)



        # 数据装载
        data_loader_train= DataLoader(dataset=train_data_label, batch_size= batch_size, shuffle=True)
        data_loader_val = DataLoader(dataset =val_data_label, batch_size = batch_size, shuffle = True)

        #改成了bce的loss（需要和sigmoid配合）
        loss_func = torch.nn.BCELoss(reduction = 'sum')
        optimizer = torch.optim.Adam(nn_model.parameters(),lr = lr, weight_decay = weight_decay_rate)
        #learning rate deacy
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= decay_step_size, gamma= decay_rate)
        #initialize early stoppping
        early_stopping = EarlyStopping(estop_patience, verbose=True)


        runing_loss_curve = []
        runing_accu_curve = []
        val_loss_curve = []
        val_accu_curve = []
        conf_matrix_set = []

        for epoch in range(num_epochs):
            print("Epoch  {}/{}".format(epoch+1, num_epochs))
            # training procedure in one epoch
            nn_model.train()
            running_loss = 0.0
            running_correct = 0.0
            for data in data_loader_train:
                X_train, Y_train = data
                X_train, Y_train = self.get_variable(X_train),self.get_variable(Y_train)
                # process output with threshold 0.5 (could change to hyperparameter if needed) to prediction
                raw_outputs = nn_model(X_train)
                output_dim = raw_outputs.ndim
                raw_outputs = raw_outputs.squeeze(output_dim-1)#squeese last dim of output since the class equal to 1
                # use sigmoid changing raw output to probability
                outputs = torch.sigmoid(raw_outputs)
                zero = torch.zeros_like(outputs)
                one = torch.ones_like(outputs)          
                pred = torch.where(outputs > 0.5, one, zero)
                running_correct += torch.sum(pred == Y_train.data)
                #initialze gradient of opt and calculate grad with loss function
                optimizer.zero_grad()
                loss = loss_func(outputs, Y_train)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            #validation procedure in one epoch
            nn_model.eval()
            val_correct = 0.0
            val_loss = 0.0
            # initialize confusion matrix
            conf_matrix = torch.zeros(num_classes, num_classes)
            for data in data_loader_val:
                X_val, Y_val = data
                X_val, Y_val = self.get_variable(X_val),self.get_variable(Y_val)
                # process output with threshold 0.5 (could change to hyperparameter if needed) to prediction
                raw_outputs = nn_model(X_val)
                output_dim = raw_outputs.ndim
                raw_outputs = raw_outputs.squeeze(output_dim-1)#squeese last dim of output since the class equal to 1

                # use sigmoid changing raw output to probability
                outputs = torch.sigmoid(raw_outputs)
                zero = torch.zeros_like(outputs)
                one = torch.ones_like(outputs)          
                pred = torch.where(outputs > 0.5, one, zero)
                val_correct += torch.sum(pred == Y_val.data)
                loss = loss_func(outputs, Y_val)
                val_loss += loss.item()

                # update confusion matrix
                if pred.ndim == 2:
                    pred = pred.contiguous().view(-1,1).squeeze(1)
                    Y_val = Y_val.contiguous().view(-1,1).squeeze(1)
                self.confusion_matrix(pred.type(torch.LongTensor), Y_val.data.type(torch.LongTensor), conf_matrix)          

            #学习率衰减
            scheduler.step()
            # store the result and print step result to the user
            conf_matrix_set.append(conf_matrix.numpy())
            runing_loss_curve.append(running_loss / train_len)
            runing_accu_curve.append(running_correct / train_len)
            val_loss_curve.append(val_loss / val_len)
            val_accu_curve.append(val_correct / val_len)
            print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Val loss is:{:.4f}, Val Accuracy is:{:.4f}%".format(
                running_loss / train_len, 100 * running_correct / train_len,
                val_loss / val_len, 100 * val_correct / val_len))     
            print("confusion matrix is:\n", conf_matrix)
            #set early stopping
            early_stopping(val_loss/val_len, val_correct/val_len, nn_model, save_path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

        # #save the model
        # torch.save(nn_model, save_path)
        del nn_model
        del data_loader_train
        del data_loader_val
        gc.collect()

        self.val_loss_min = early_stopping.val_loss_min
        self.val_accu_max = early_stopping.val_accu
        self.running_loss_curve = runing_accu_curve
        self.running_accu_curve = runing_accu_curve
        self.val_loss_curve = val_loss_curve
        self.val_accu_curve = val_accu_curve
        self.val_conf_mat = np.asarray(conf_matrix_set)
        print('val_loss_min:', early_stopping.val_loss_min)
        print('corresponding val_accu', early_stopping.val_accu)
        return runing_loss_curve, runing_accu_curve, val_loss_curve, val_accu_curve, np.asarray(conf_matrix_set)




    #nn evaluation function
    def eval_nn(self, test_data_label, model_path):
        '''This function is to test the crnn performance'''
        test_len = test_data_label.segment_len

        batch_size = self.cfg.batch_size
        num_classes = self.cfg.num_classes

        # 加载神经网络
        nn_model = torch.load(model_path)
        nn_model = self.change_device(nn_model)
        nn_model.eval()

        # load test data
        data_loader_test = DataLoader(dataset =test_data_label, batch_size = batch_size, shuffle = False)
        loss_func = torch.nn.BCELoss(reduction = 'sum')

        testing_correct = 0.0
        testing_loss = 0.0        
        # initialize confusion matrix
        conf_matrix = torch.zeros(num_classes, num_classes)
        with torch.no_grad():
            for data in data_loader_test:
                X_test, Y_test = data
                X_test, Y_test = self.get_variable(X_test),self.get_variable(Y_test)
                # process output with threshold 0.5 (could change to hyperparameter if needed) to prediction
                raw_outputs = nn_model(X_test)
                output_dim = raw_outputs.ndim
                raw_outputs = raw_outputs.squeeze(output_dim-1)#squeese last dim of output since the class equal to 1
                outputs = torch.sigmoid(raw_outputs)
                zero = torch.zeros_like(outputs)
                one = torch.ones_like(outputs)          
                pred = torch.where(outputs > 0.5, one, zero)
                testing_correct += torch.sum(pred == Y_test.data)
                # calculate BCE loss with sigmoided value
                loss = loss_func(outputs, Y_test)
                testing_loss += loss.item()
                # update confusion matrix
                if pred.ndim == 2:
                    pred = pred.contiguous().view(-1,1).squeeze(1)
                    Y_test = Y_test.contiguous().view(-1,1).squeeze(1)
                self.confusion_matrix(pred.type(torch.LongTensor), Y_test.data.type(torch.LongTensor), conf_matrix)          
        
        del nn_model
        del data_loader_test
        gc.collect()

        self.test_loss = testing_loss/test_len
        self.test_accu = testing_correct/test_len
        self.test_conf_mat = conf_matrix.numpy()
        return testing_loss/test_len, testing_correct/test_len, conf_matrix.numpy()



    def predict_nn(self, predic_data, predict_type, model_path):
        '''This function is to get the one-hot output from CRNN'''

        batch_size = self.cfg.batch_size

        # 加载神经网络
        nn_model = torch.load(model_path)
        nn_model = self.change_device(nn_model)
        nn_model.eval()

        # load the predict data
        predic_loader = DataLoader(dataset=predic_data, batch_size= batch_size, shuffle=False)


        batch_output = []
        batch_label = []
        with torch.no_grad():
            for data in predic_loader:
                X_predict, Y_predict = data
                X_predict, Y_predict = self.get_variable(X_predict),self.get_variable(Y_predict)

                raw_outputs = nn_model(X_predict)
                output_dim = raw_outputs.ndim
                raw_outputs = raw_outputs.squeeze(output_dim-1)#squeese last dim of output since the class equal to 1
                prob_outputs = torch.sigmoid(raw_outputs)
                zero = torch.zeros_like(prob_outputs)
                one = torch.ones_like(prob_outputs)          
                onehot_outputs = torch.where(prob_outputs > 0.5, one, zero)

                #change all variable to 1 dimension
                if raw_outputs.ndim == 2:
                    raw_outputs = raw_outputs.contiguous().view(-1,1).squeeze(1)
                    prob_outputs = prob_outputs.contiguous().view(-1,1).squeeze(1)
                    onehot_outputs = onehot_outputs.contiguous().view(-1,1).squeeze(1)
                    Y_predict = Y_predict.contiguous().view(-1,1).squeeze(1)
                    
                raw_outputs = raw_outputs.cpu().numpy()
                # print(raw_outputs.shape)
                prob_outputs = prob_outputs.cpu().numpy()
                onehot_outputs = onehot_outputs.cpu().numpy().astype(np.int16)
                Y_predict = Y_predict.cpu().numpy().astype(np.int16)


                if predict_type == 'raw':
                    batch_output.extend(raw_outputs)
                elif predict_type == 'prob':
                    batch_output.extend(prob_outputs)
                elif predict_type == 'onehot':
                    batch_output.extend(onehot_outputs)
                batch_label.extend(Y_predict)

        batch_output = np.asarray(batch_output)
        batch_label = np.asarray(batch_label)
        # print('batch label shape',batch_label.shape)
        # print('batch pred shape', batch_output.shape)
        batch_pred_label = {'pred': batch_output, 'label': batch_label}

        del predic_loader       
        gc.collect()
        return batch_pred_label






class model_predict(DL_process):
    def __init__(self, cfg, pred_seq_len, seed, model_path):
        super().__init__(cfg, seed)
        self.model_path = model_path
        self.pred_seq_len = pred_seq_len
        self.train_raw_predict = None
        self.train_prob_predict = None
        self.train_bino_predict = None
        self.val_raw_predict = None
        self.val_prob_predict = None
        self.val_bino_predict = None
        self.test_raw_predict = None
        self.test_prob_predict = None
        self.test_bino_predict = None


    def train_val_predict(self, train_set, val_set):
        self.train_raw_predict = super().predict_nn(train_set, 'raw', self.model_path)
        self.train_raw_predict['pred'] = self.train_raw_predict['pred'].reshape((-1, self.pred_seq_len))
        self.train_raw_predict['label'] = self.train_raw_predict['label'].reshape((-1, self.pred_seq_len))

        self.train_prob_predict = super().predict_nn(train_set, 'prob', self.model_path)
        self.train_prob_predict['pred'] = self.train_prob_predict['pred'].reshape((-1, self.pred_seq_len))
        self.train_prob_predict['label'] = self.train_prob_predict['label'].reshape((-1, self.pred_seq_len))

        self.train_bino_predict = super().predict_nn(train_set, 'onehot', self.model_path)
        self.train_bino_predict['pred'] = self.train_bino_predict['pred'].reshape((-1, self.pred_seq_len))
        self.train_bino_predict['label'] = self.train_bino_predict['label'].reshape((-1, self.pred_seq_len))

        self.val_raw_predict = super().predict_nn(val_set, 'raw', self.model_path)
        self.val_raw_predict['pred'] = self.val_raw_predict['pred'].reshape((-1, self.pred_seq_len))
        self.val_raw_predict['label'] = self.val_raw_predict['label'].reshape((-1, self.pred_seq_len))

        self.val_prob_predict = super().predict_nn(val_set, 'prob', self.model_path)
        self.val_prob_predict['pred'] = self.val_prob_predict['pred'].reshape((-1, self.pred_seq_len))
        self.val_prob_predict['label'] = self.val_prob_predict['label'].reshape((-1, self.pred_seq_len))

        self.val_bino_predict = super().predict_nn(val_set, 'onehot', self.model_path)
        self.val_bino_predict['pred'] = self.val_bino_predict['pred'].reshape((-1, self.pred_seq_len))
        self.val_bino_predict['label'] = self.val_bino_predict['label'].reshape((-1, self.pred_seq_len))




    def test_predict(self, test_set):
        self.test_raw_predict = super().predict_nn(test_set, 'raw', self.model_path)
        self.test_raw_predict['pred'] =  self.test_raw_predict['pred'].reshape((-1, self.pred_seq_len))
        self.test_raw_predict['label'] = self.test_raw_predict['label'].reshape((-1, self.pred_seq_len))

        self.test_prob_predict = super().predict_nn(test_set, 'prob', self.model_path)
        self.test_prob_predict['pred'] = self.test_prob_predict['pred'].reshape((-1, self.pred_seq_len))
        self.test_prob_predict['label'] = self.test_prob_predict['label'].reshape((-1, self.pred_seq_len))

        self.test_bino_predict = super().predict_nn(test_set, 'onehot', self.model_path)
        self.test_bino_predict['pred'] = self.test_bino_predict['pred'].reshape((-1, self.pred_seq_len))
        # print(self.test_bino_predict['pred'].shape)
        self.test_bino_predict['label'] = self.test_bino_predict['label'].reshape((-1, self.pred_seq_len))
        # print(self.test_bino_predict['label'].shape)