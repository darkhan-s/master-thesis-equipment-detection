from ast import arg
from cProfile import label
from cgitb import enable
import json
from selectors import EpollSelector
from turtle import color
from click import style
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import argparse
import numpy as np
from pyparsing import alphas
import matplotlib.colors as mcolors

FONT_SIZE = 13    
COLORS = ["blue", "darkcyan", "darkgreen", "purple",  "darkorange", "black"]

### Ideas borrowed from https://github.com/facebookresearch/detectron2/issues/3180

# average AP50 of multiple continual learning results
def average_out(metrics, main_metric_name):

    shortest_iters = len(metrics[0].AP50_val)
    iters_val_AP50 = None

    for metric in metrics: 
        shortest_iters = min(shortest_iters, len(metric.AP50_val))
    total_ap = np.zeros(shortest_iters)
    for metric in metrics:        
        iters_val_AP50 = [x['iteration'] for x in metric.experiment_metrics if (main_metric_name + metric.text_to_append) in x]
        AP50_val = [x[main_metric_name + metric.text_to_append] for x in metric.experiment_metrics if (main_metric_name + metric.text_to_append) in x]
        for i in range(shortest_iters):
            total_ap[i] = total_ap[i] + AP50_val[i]

    for i in range(shortest_iters):
        total_ap[i] = total_ap[i] / len(metrics)
    total_iters = iters_val_AP50[:shortest_iters]
    return total_ap, total_iters

class Metrics:
    def __init__(self, path, main_metric_name, class_id = ''):
        self.text_to_append = ''
        self.class_id = class_id
        self.read_loss_metrics(path, main_metric_name)
        self.read_LR_metrics(path)
        self.read_constLoss(path)
        self.experiment_metrics = self.load_json_arr(path + '/metrics.json')

    ## function to read metrics from json
    def load_json_arr(self, json_path):
            lines = []
            with open(json_path, 'r') as f:
                for line in f:
                    lines.append(json.loads(line))
            return lines

    def read_loss_metrics(self, path, main_metric_name):
        experiment_metrics = self.load_json_arr(path + '/metrics.json')

        if self.class_id is not None and self.class_id != 0:
            self.text_to_append = text_to_append = '-Model '+ str(self.class_id)
        elif self.class_id == 0:
            self.text_to_append = text_to_append = ''
        
        ## read total loss
        self.iters_total_loss = [x['iteration'] for x in experiment_metrics if 'total_loss' in x]
        self.total_loss = [x['total_loss'] for x in experiment_metrics if 'total_loss' in x]

        ## read AP50 for the teacher model (which accepts only target images)   
        self.iters_val_AP50 = [x['iteration'] for x in experiment_metrics if (main_metric_name + text_to_append) in x]
        self.AP50_val = [x[main_metric_name + text_to_append] for x in experiment_metrics if (main_metric_name + text_to_append) in x]
        ## read AP50 for the student model (which accepts both source and target images)   
        self.iters_ap50 = [x['iteration'] for x in experiment_metrics if ('bbox_student/AP50' + text_to_append) in x]
        self.ap50 = [x['bbox_student/AP50' + text_to_append] for x in experiment_metrics if ('bbox_student/AP50' + text_to_append) in x]

    def read_LR_metrics(self, path): 
        experiment_metrics = self.load_json_arr(path + '/metrics.json')
        ## read LR for the ensembled model   
        self.iters = [x['iteration'] for x in experiment_metrics if 'lr' in x]
        self.lr = [x['lr'] for x in experiment_metrics if 'lr' in x]

    def read_constLoss(self, path): 
        experiment_metrics = self.load_json_arr(path + '/metrics.json')
        ## read consistency losses for the ensembled model   
        self.iters_const_s = [x['iteration'] for x in experiment_metrics if 'loss_consistency_t' in x]
        self.iters_const_t = [x['iteration'] for x in experiment_metrics if 'loss_consistency_s' in x]
        self.const_loss_t = [x['loss_consistency_t'] for x in experiment_metrics if 'loss_consistency_t' in x]
        self.const_loss_s = [x['loss_consistency_s'] for x in experiment_metrics if 'loss_consistency_s' in x]
        self.ins_cls_s = [x['loss_DA_ins_cls'] for x in experiment_metrics if 'loss_DA_ins_cls' in x]
        self.ins_cls_t = [x['loss_tgt_DA_ins_cls'] for x in experiment_metrics if 'loss_tgt_DA_ins_cls' in x]
        
        self.const_total =  [x['loss_consistency_t'] + x['loss_consistency_s'] for x in experiment_metrics if 'loss_consistency_t' in x and 'loss_consistency_s' in x]
        self.iters_total =  [x['iteration'] for x in experiment_metrics if 'loss_consistency_t' in x]
        self.ins_cls_total = [x['loss_DA_ins_cls'] + x['loss_tgt_DA_ins_cls'] for x in experiment_metrics if 'loss_DA_ins_cls' in x and 'loss_tgt_DA_ins_cls' in x]


def plot_two(metrics, enable_LR_plot = False, preffered_names = ['']):

    metrics_original = metrics[0]
    filename_toSave = '/loss&AP50.jpg'
    fig, ax = plt.subplots(2,1, figsize=(14,8))
    if enable_LR_plot:
        fig, ax = plt.subplots(3,1, figsize=(14,8))
    
    offset = (-5, np.random.randint(-100,0))

    ## plot the AP values for teacher model
    best_iter_val = metrics_original.AP50_val.index(max(metrics_original.AP50_val))
    ax[0].plot(metrics_original.iters_val_AP50, metrics_original.AP50_val, color="red", label = 'Teacher ' + preffered_names[0], linewidth=2.0)
    ax[0].vlines(metrics_original.iters_val_AP50[best_iter_val], -5, 100,color="red", ls='--', label = None,transform=ax[0].get_xaxis_transform())
    ax[0].annotate('Max AP50, teacher: %.2f %% at iter: %d'%(float(max(metrics_original.AP50_val)), int(metrics_original.iters_val_AP50[best_iter_val])), xy=(metrics_original.iters_val_AP50[best_iter_val], max(metrics_original.AP50_val)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='red') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    ax[0].set_xlim([-1000,max(max(metrics_original.iters_total_loss),max(metrics_original.iters_val_AP50))])
    ax[0].set_ylim([-5,max(1.1*max(metrics_original.total_loss),1.1*max(metrics_original.AP50_val))])

    offset = (-5, np.random.randint(-100,0))

    ax[0].plot(metrics_original.iters_ap50, metrics_original.ap50, color="blue", label = 'Student ' + preffered_names[0],linewidth=2.0, alpha = 0.3)
    ax[0].set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax[0].set_ylabel('AP50(%)', fontsize=FONT_SIZE)
    best_iter = metrics_original.ap50.index(max(metrics_original.ap50))
    ax[0].vlines(metrics_original.iters_ap50[best_iter], -5, 100 ,color="blue", ls='--', label = None)
    ax[0].annotate('Max AP50, student: %.2f %% at iter: %d'%(float(max(metrics_original.ap50)),int(metrics_original.iters_ap50[best_iter])),xy=(metrics_original.iters_ap50[best_iter],max(metrics_original.ap50)),xytext=offset,textcoords='offset points',fontsize=FONT_SIZE, color='blue') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    ax[0].set_xlim([-1000,max(max(metrics_original.iters_total_loss),max(metrics_original.iters_val_AP50))])
    ax[0].set_ylim([-5,100])
    
    ## plot the total loss
    ax[1].plot(metrics_original.iters_total_loss, metrics_original.total_loss, color = 'r', linewidth=0.8, label = 'Teacher model')
    ax[1].set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax[1].set_ylabel('Total loss', fontsize=FONT_SIZE)

    ax[1].set_xlim([-1000,max(max(metrics_original.iters_total_loss),max(metrics_original.iters_val_AP50))])
    # ax[1].set_ylim([min(total_loss), max(total_loss)])
    ## extend the max dashlines to first subplot
    ax[1].vlines(metrics_original.iters_ap50[best_iter], 0, 1, color="blue", ls='--', transform=ax[1].get_xaxis_transform())
    ax[1].vlines(metrics_original.iters_val_AP50[best_iter_val], 0, 1, color="red", ls='--', transform=ax[1].get_xaxis_transform())

    if enable_LR_plot:
        ## plot the LR
        ax[2].plot(metrics_original.iters, metrics_original.lr, color="red", label = 'Scheduler',linewidth=1.5)
        #ax[2].plot(metrics_extra.iters, metrics_extra.lr, color="blue", label = 'New scheduler',linewidth=1.5)
        ax[2].legend(loc='best', title="Scheduler", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)    
        ax[2].set_xlabel('Iterations', fontsize=FONT_SIZE)
        ax[2].set_ylabel('Learning rate', fontsize=FONT_SIZE)
        ax[2].legend(loc='best', title="Scheduler", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1) 
        ax[2].grid()      
        ax[2].legend(loc='best', title="AP50", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   

        filename_toSave = '/loss_AP50_LR.jpg'



    ax[0].tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ax[1].tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)

    ## set up the grid and the legend
    ax[0].legend(loc='best', title="AP50", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   
    ax[1].legend(loc='best', title="Total loss", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)     

    ax[0].grid()
    ax[1].grid()
    
    plt.tight_layout()
    # plt.show()
    path = args.path[0] + filename_toSave
    plt.savefig(path)
    plt.close()



def plot_consistency(metrics, preffered_names = ['']):
    import string

    metrics_original = metrics[0]
    filename_toSave = '/consistency_loss.jpg'
    fig, ax = plt.subplots(2,1, figsize=(14,8))
    
    letter = string.ascii_lowercase[0]

    ## plot the loss term
    ax[0].plot(metrics_original.iters_total, metrics_original.const_total, color="red", label = letter + ') Consistency loss ' + preffered_names[0], linewidth=1.5)
    ax[0].set_xlim([-1000,max(max(metrics_original.const_total),max(metrics_original.ins_cls_total))])
    ax[0].set_ylim([-5,max(1.1*max(metrics_original.ins_cls_total),1.1*max(metrics_original.const_total))])
    ax[0].set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax[0].set_ylabel('Consistency loss', fontsize=FONT_SIZE)
    
    #plot another loss term
    ax[1].plot(metrics_original.iters_total, metrics_original.ins_cls_total, color = 'r', linewidth=1.5, label = letter + ') Instance-level loss ' + preffered_names[0])
    ax[1].set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax[1].set_ylabel('Instance-level loss', fontsize=FONT_SIZE)

    if len(metrics)>1 and len(metrics)<=len(COLORS) and len(metrics)<=len(string.ascii_lowercase):
            metrics = metrics[1:]

            if len(preffered_names)<len(metrics):
                for j in range(len(metrics)-len(preffered_names)):
                    preffered_names.append('')
                    print("Check the names..")

            for i in range(len(metrics)):
                metrics_extra = metrics[i]
                letter = string.ascii_lowercase[i+1]

                ax[0].plot(metrics_extra.iters_total, metrics_extra.const_total, color=COLORS[i], label = letter + ') Consistency loss ' + preffered_names[i+1], linewidth=1.5)
                ax[0].set_ylim([0,0.6])
                ax[0].set_xlim([-1000,max(max(metrics_extra.iters_total),max(metrics_extra.const_total))])

                ax[1].plot(metrics_extra.iters_total, metrics_extra.ins_cls_total, color = COLORS[i], linewidth=1.5, label = letter + ') Instance-level loss ' + preffered_names[i+1])
                ax[1].set_xlim([-1000,max(max(metrics_extra.iters_total),max(metrics_extra.const_total))])
    else:
        print("Limited number of plots is possible, add more colors or letters..")

    ax[0].tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ax[1].tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ## set up the grid and the legend
    ax[0].legend(loc='lower right', title="Consistency loss", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)  
    ax[1].legend(loc='lower right', title="Instance-level loss", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   

    ax[0].grid()
    ax[1].grid()

    plt.tight_layout()
    # plt.show()
    path = args.path[0] + filename_toSave
    plt.savefig(path)
    plt.close()
    

def plot_only_ap_multi(metrics, preffered_names = [''], disable_student = False, title = "AP50"):

    metrics_original = metrics[0]
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    ## plot the AP values for teacher model
    best_iter_val = metrics_original.AP50_val.index(max(metrics_original.AP50_val))
    ax.plot(metrics_original.iters_val_AP50, metrics_original.AP50_val, color="red", label = 'a) Teacher model ' + preffered_names[0], linewidth=1.5)
    ax.vlines(metrics_original.iters_val_AP50[best_iter_val], -5, 100,color="red", ls='--', label = None,transform=ax.get_xaxis_transform())
    offset = (5, np.random.randint(-200,200))
    ax.annotate('a) Max AP50, teacher: %.2f %% at iter: %d'%(float(max(metrics_original.AP50_val)), int(metrics_original.iters_val_AP50[best_iter_val])), xy=(metrics_original.iters_val_AP50[best_iter_val], max(metrics_original.AP50_val)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='red') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    ax.set_xlim([-1000,max(max(metrics_original.iters_total_loss),max(metrics_original.iters_val_AP50))])
    #ax.set_ylim([-5,max(1.1*max(metrics_original.total_loss),1.1*max(metrics_original.AP50_val))])

    

    if not disable_student:
        ax.plot(metrics_original.iters_ap50, metrics_original.ap50, color="red", label = 'a) Student model '+ preffered_names[0],linewidth=2.5, alpha = 0.3)
        #best_iter = metrics_original.ap50.index(max(metrics_original.ap50))
        #ax.vlines(metrics_original.iters_ap50[best_iter], -5, 100 ,color="red", ls='--', label = None, alpha=0.3)
        #ax.annotate('a) Max AP50, student: %f %% at iter: %d'%(float(max(metrics_original.ap50)),int(metrics_original.iters_ap50[best_iter])),xy=(metrics_original.iters_ap50[best_iter],max(metrics_original.ap50)),xytext=(5,-45),textcoords='offset points',fontsize=FONT_SIZE, color='red', alpha=0.7) #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
        ax.set_xlim([-1000,max(metrics_original.iters_ap50)])
        ax.set_ylim([-5,100])
    
    ax.set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax.set_ylabel('AP50(%)', fontsize=FONT_SIZE)
    ## need to change this if decide to use more plots in one graph
    # linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']
    # #  ('solid', 'solid'),      # Same as (0, ()) or '-'
    # #  ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    # #  ('dashed', 'dashed'),    # Same as '--'
    # #  ('dashdot', 'dashdot')]  # Same as '-.'
    import string 
    ## append if not none
    if len(metrics)>1 and len(metrics)<=len(COLORS) and len(metrics)<=len(string.ascii_lowercase):
        metrics = metrics[1:]
        if len(preffered_names)<len(metrics):
            for j in range(len(metrics)-len(preffered_names)):
                preffered_names.append('')
                print("Check the names..")

        for i in range(len(metrics)):
            metrics_extra = metrics[i]
            letter = string.ascii_lowercase[i+1]

            best_iter_val = metrics_extra.AP50_val.index(max(metrics_extra.AP50_val))
            ax.plot(metrics_extra.iters_val_AP50, metrics_extra.AP50_val, color=COLORS[i], label = letter + ") Teacher model " + preffered_names[i+1], linewidth=1.5)
            
            if not disable_student:
                ax.plot(metrics_extra.iters_ap50, metrics_extra.ap50, color=COLORS[i], label = letter + ") Student model " + preffered_names[i+1], linewidth=1.5, alpha = 0.3)

            ax.vlines(metrics_extra.iters_val_AP50[best_iter_val], -5, 100,color=COLORS[i], ls='--', label = None,transform=ax.get_xaxis_transform())
            
            # if the annotation does not fit into the frame
            ax.xlim = ax.get_xlim()
            offset = (5, np.random.randint(-200,200))
            if metrics_extra.iters_val_AP50[best_iter_val] > ax.xlim[1]:
                offset = (-250, np.random.randint(-200,200)) 
            ax.annotate(letter + ") Max AP50, teacher: %.2f %% at iter: %d"%(float(max(metrics_extra.AP50_val)), int(metrics_extra.iters_val_AP50[best_iter_val])), xy=(metrics_extra.iters_val_AP50[best_iter_val], max(metrics_extra.AP50_val)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color=COLORS[i]) #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')

            if metrics_original.iters_ap50 < metrics_extra.iters_ap50:
                ax.set_xlim([-1000,max(metrics_original.iters_ap50)])
            if metrics_original.iters_val_AP50 < metrics_extra.iters_val_AP50:
                ax.set_xlim([-1000,max(max(metrics_extra.iters_total_loss),max(metrics_extra.iters_val_AP50))])
                #ax.set_ylim([-5,max(1.1*max(metrics_extra.total_loss),1.1*max(metrics_extra.AP50_val))])
                ax.set_ylim([-5,100])


    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ## set up the grid and the legend
    ax.legend(loc='best', title=title, fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)    

    ax.grid()

    plt.tight_layout()
    # plt.show()
    path = args.path[0] + '/AP50.jpg'
    plt.savefig(path)
    plt.close()

def plot_only_lr_multi(metrics):
    metrics_original = metrics[0]
    fig, ax = plt.subplots(1,1, figsize=(14,8))

    ## plot the AP values for teacher model
    ax.plot(metrics_original.iters, metrics_original.lr, color="red", label = 'a) Original scheduler', linewidth=4)
    ax.set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax.set_ylabel('Learning rate', fontsize=FONT_SIZE)
    
    ## append if not none
    if len(metrics)==2:
        metrics_extra = metrics[1]
        ax.plot(metrics_extra.iters, metrics_extra.lr, color="blue", label = 'b) Scheduler with cosine annealing (without restart)', linewidth=2, linestyle='--')
        
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ## set up the grid and the legend
    ax.legend(loc='best', title="Learning rate schedulers", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   

    ax.grid()
    plt.tight_layout()
    # plt.show()
    path = args.path[0] + '/LR.jpg'
    plt.savefig(path)
    plt.close()

def plot_average_ap(metrics, metrics_hardcoded, main_metrics, metrics_hardcoded_single):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    total_ap, total_iters = average_out(metrics, main_metrics)
    total_ap_hardcoded, total_iters_hardcoded = average_out(metrics_hardcoded, main_metrics)
    total_ap_hardcoded_single, total_iters_hardcoded_single = average_out(metrics_hardcoded_single, main_metrics)


    ## plot the AP values for teacher model
    ax.plot(total_iters, total_ap, color="red", label = 'a) Average continual AP50 for 10 classes (21-30)', linewidth=1.5)
    ax.plot(total_iters_hardcoded, total_ap_hardcoded, color="blue", label = 'b) Average AP50 for 10 classes (21-30), model trained on classes 1-30', linewidth=1.5)
    ax.plot(total_iters_hardcoded_single, total_ap_hardcoded_single, color="darkgreen", label = 'c) Average AP50 for 10 classes (21-30) trained only on the single class data ', linewidth=1.5)

    ax.set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax.set_ylabel('AP50%', fontsize=FONT_SIZE)
    
    best_iter_1 = int((np.where(total_ap == max(total_ap)))[0])
    ax.vlines(total_iters[best_iter_1], -5, 100,color="red", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if total_iters[best_iter_1] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('a' + ") Max AP50: %.2f %% at iter: %d"%(float(max(total_ap)), int(total_iters[best_iter_1])), xy=(total_iters[best_iter_1], max(total_ap)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='red') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')

    best_iter_2 = int((np.where(total_ap_hardcoded == max(total_ap_hardcoded)))[0])
    ax.vlines(total_iters_hardcoded[best_iter_2], -5, 100,color="blue", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    best_iter_3 = int((np.where(total_ap_hardcoded_single == max(total_ap_hardcoded_single)))[0])
    ax.vlines(total_iters_hardcoded_single[best_iter_3], -5, 100,color="darkgreen", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if total_iters_hardcoded[best_iter_2] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('b' + ") Max AP50: %.2f %% at iter: %d"%(float(max(total_ap_hardcoded)), int(total_iters_hardcoded[best_iter_2])), xy=(total_iters_hardcoded[best_iter_2], max(total_ap_hardcoded)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='blue') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if total_iters_hardcoded_single[best_iter_3] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('c' + ") Max AP50: %.2f %% at iter: %d"%(float(max(total_ap_hardcoded_single)), int(total_iters_hardcoded_single[best_iter_3])), xy=(total_iters_hardcoded_single[best_iter_3], max(total_ap_hardcoded_single)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='darkgreen') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ## set up the grid and the legend
    ax.legend(loc='best', title="Continual AP50%", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   

    ax.grid()
    plt.tight_layout()
    # plt.show()
    path = args.path[0] + '/continualAP_average.jpg'
    plt.savefig(path)
    plt.close()

def plot_for_group_of_classes(metrics, metrics_to_compare, main_metrics):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    total_ap, total_iters = average_out(metrics, main_metrics)
    total_ap_hardcoded, total_iters_hardcoded = average_out(metrics_to_compare, main_metrics)

    ## plot the AP values for teacher model
    ax.plot(total_iters, total_ap, color="red", label = 'a) Average AP50 for classes 1-4', linewidth=1.5)
    ax.plot(total_iters_hardcoded, total_ap_hardcoded, color="blue", label = 'b) Average AP50 for classes 5-30', linewidth=1.5)

    ax.set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax.set_ylabel('AP50%', fontsize=FONT_SIZE)
    
    best_iter_1 = int((np.where(total_ap == max(total_ap)))[0])
    ax.vlines(total_iters[best_iter_1], -5, 100,color="red", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if total_iters[best_iter_1] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('a' + ") Max AP50: %.2f %% at iter: %d"%(float(max(total_ap)), int(total_iters[best_iter_1])), xy=(total_iters[best_iter_1], max(total_ap)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='red') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')

    best_iter_2 = int((np.where(total_ap_hardcoded == max(total_ap_hardcoded)))[0])
    ax.vlines(total_iters_hardcoded[best_iter_2], -5, 100,color="blue", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if total_iters_hardcoded[best_iter_2] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('b' + ") Max AP50: %.2f %% at iter: %d"%(float(max(total_ap_hardcoded)), int(total_iters_hardcoded[best_iter_2])), xy=(total_iters_hardcoded[best_iter_2], max(total_ap_hardcoded)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='blue') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ## set up the grid and the legend
    ax.legend(loc='best', title="AP50%", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   

    ax.grid()
    plt.tight_layout()
    # plt.show()
    path = args.path[0] + '/AP50_per_class_group.jpg'
    plt.savefig(path)
    plt.close()


def plot_to_compare_21(metrics, metrics_to_compare, main_metrics):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    metrics = metrics[0]
    #total_ap, total_iters = average_out(metrics, main_metrics)
    total_ap_hardcoded, total_iters_hardcoded = average_out(metrics_to_compare, main_metrics)

    ## plot the AP values for teacher model
    ax.plot(metrics.iters_val_AP50, metrics.AP50_val, color="red", label = 'a) Continual learning on class 21 (AP50 for classes 1-21)', linewidth=1.5)
    ax.plot(total_iters_hardcoded, total_ap_hardcoded, color="blue", label = 'b) Training on full dataset 1-30 (average result for classes 1-21)', linewidth=1.5)

    ax.set_xlabel('Iterations', fontsize=FONT_SIZE)
    ax.set_ylabel('AP50%', fontsize=FONT_SIZE)
    best_iter_1 = metrics.AP50_val.index(max(metrics.AP50_val))
    ax.vlines(metrics.iters_val_AP50[best_iter_1], -5, 100,color="red", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if metrics.iters_val_AP50[best_iter_1] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('a' + ") Max AP50: %.2f %% at iter: %d"%(float(max(metrics.AP50_val)), int(metrics.iters_val_AP50[best_iter_1])), xy=(metrics.iters_val_AP50[best_iter_1], max(metrics.AP50_val)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='red') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')

    best_iter_2 = int((np.where(total_ap_hardcoded == max(total_ap_hardcoded)))[0])
    ax.vlines(total_iters_hardcoded[best_iter_2], -5, 100,color="blue", ls='--', label = None,transform=ax.get_xaxis_transform())
        
    # if the annotation does not fit into the frame
    ax.xlim = ax.get_xlim()
    offset = (-5, np.random.randint(-200,200))
    if total_iters_hardcoded[best_iter_2] > ax.xlim[1]:
        offset = (-250, np.random.randint(-200,200)) 
    ax.annotate('b' + ") Max AP50: %.2f %% at iter: %d"%(float(max(total_ap_hardcoded)), int(total_iters_hardcoded[best_iter_2])), xy=(total_iters_hardcoded[best_iter_2], max(total_ap_hardcoded)), xytext=offset, textcoords='offset points', fontsize=FONT_SIZE, color='blue') #arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
    ## set up the grid and the legend
    ax.legend(loc='best', title="AP50%", fancybox=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)   

    ax.grid()
    plt.tight_layout()
    # plt.show()
    path = args.path[0] + '/AP50_continual_total_for21.jpg'
    plt.savefig(path)
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, action='append',
                        help='Training output, full path to metrics.json')
    parser.add_argument('--path_name', type=str, action='append',
                        help='Short name for the path to use in the legend')
    parser.add_argument('--class_id', type=int, action='append')
    parser.add_argument('--disable_student', action='store_true')
    parser.add_argument('--title', type=str, default = "AP50")
    parser.add_argument('--metric_name', type=str, default = 'bbox_teacher/AP50')
    # fix later: 
    # parser.add_argument('--save_dest', type=str)
    # parser.add_argument('--save_name_postfix', type=str)

    args = parser.parse_args()
    metrics = []
    for i in range(len(args.path)):
        metrics.append(Metrics(args.path[i], class_id=args.class_id[i], main_metric_name=args.metric_name))

    ## manually set names for the plots 
    ##names = ['(No modifications)', '(New scheduler)']
    names = [f"({name})" for name in args.path_name]

    for i in range(len(names)): 
        names[i] = names[i].replace("_", " ")
    
    title = args.title.replace("_", " ")

    plot_only_ap_multi(metrics, preffered_names = names, disable_student=args.disable_student, title = title)
    plot_two(metrics, preffered_names = names, enable_LR_plot = False)
    plot_two(metrics, preffered_names = names, enable_LR_plot = True)
    plot_only_lr_multi(metrics)
    plot_consistency (metrics,preffered_names = names)

    if  args.metric_name == 'bbox_teacher/AP50':
        metrics_hardCoded = []
        for i in range(10):
            metrics_hardCoded.append(Metrics("output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1", class_id=21+i, main_metric_name=args.metric_name))
        metrics_hardcoded_single = []
       
        for i in range(10):
            name = f"output-mymodel-classes-{21+i}only-FINAL-MyModel_withCustomAugmentation"

            # temp fix
            if i == 2:
                metrics_hardcoded_single.append(Metrics(name, class_id=23, main_metric_name=args.metric_name))
            else:
                metrics_hardcoded_single.append(Metrics(name, class_id=0, main_metric_name=args.metric_name))
        
        plot_average_ap(metrics, metrics_hardCoded, args.metric_name, metrics_hardcoded_single)

        metrics_1_30 = []
        for i in range(21):
            metrics_1_30.append(Metrics("output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-1", class_id=1+i, main_metric_name=args.metric_name))
        #plot_to_compare_21(metrics, metrics_1_30, main_metrics=args.metric_name)

    metrics_four_classes = []
    metrics_the_rest = []
    name = f"output-mymodel-classes-1-30-FINAL-MyModel_withCustomAugmentation-origScheduler"

    for i in range(4):
        metrics_four_classes.append(Metrics(name, class_id=1+i, main_metric_name=args.metric_name))
    for i in range(25):
        metrics_the_rest.append(Metrics(name, class_id=5+i, main_metric_name=args.metric_name))
    
    plot_for_group_of_classes(metrics_four_classes, metrics_the_rest, args.metric_name)
