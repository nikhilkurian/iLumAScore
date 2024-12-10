import yaml


with open(r'/media/nikhil_iitb/extra_storage/Nikhil/lobular/codes2/config.yaml') as file:
    args = yaml.load(file, Loader=yaml.FullLoader)

args['model_name']='_loss:'+str(args['loss_criterion'])+'_wd:'+str(args['weight_decay'])+'_lr:'+str(args['lr'])