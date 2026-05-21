import torch
from AIPUBuilder.core import *
from AIPUBuilder.Optimizer.register import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
from math import sqrt as sqrt
from itertools import product as product
import math

from collections import namedtuple
from collections import defaultdict
from AIPUBuilder.Optimizer.utils import *


register_optype("DetectionOutput")

@op_register(OpType.DetectionOutput,2)
def DetectionOutput(self,*args):
    num_classes_=2
    share_location=True
    background_label_id=0
    code_type_='CENTER_SIZE'
    keep_top_k_=750
    
    num_loc_classes_=1
    num_pred_per_class=21824
    num_priors=21824
    variance_encoded_in_target=0
    #nms_param 0.05
    confidence_threshold_=0.08
    nms_threshold=0.3
    top_k=1000
    eta=1
    
    loc_data_=self.inputs[0].betensor#[1,1,87296]->[1,87296]
    conf_data_=self.inputs[1].betensor#[1,43648]
    prior_data_=self.inputs[2].betensor#[1,87296,2]

    loc_data=loc_data_.view(1,87296).permute(1,0)
    conf_data=conf_data_.permute(1,0)
    prior_data=prior_data_[:,:,0].view(-1)
    prior_variances_data=prior_data_[:,:,1].view(-1)

    all_loc_preds = torch.zeros([num_pred_per_class,num_classes_,4])
    for p in range(0,num_pred_per_class):
        start_idx = p * num_loc_classes_ * 4
        for c in range(0,num_loc_classes_):
            label = c
            all_loc_preds[p][label][0]=loc_data[start_idx + c * 4]#xmin
            all_loc_preds[p][label][1]=loc_data[start_idx + c * 4 + 1]#ymin
            all_loc_preds[p][label][2]=loc_data[start_idx + c * 4 + 2]#xmax
            all_loc_preds[p][label][3]=loc_data[start_idx + c * 4 + 3]#ymax

    all_conf_scores = torch.zeros([num_classes_,num_pred_per_class])
    for p in range(0,num_pred_per_class):
      start_idx = p * num_classes_
      for c in range(0,num_classes_):
        all_conf_scores[c][p]=conf_data[start_idx + c]

    prior_bboxes = torch.zeros([num_priors,4])
    prior_variances = torch.zeros([num_priors,4])
    for i in range(0,num_priors):
        start_idx = i * 4
        prior_bboxes[i][0]=prior_data[start_idx]#xmin
        prior_bboxes[i][1]=prior_data[start_idx + 1]#ymin
        prior_bboxes[i][2]=prior_data[start_idx + 2]#xmax
        prior_bboxes[i][3]=prior_data[start_idx + 3]#ymax

    for i in range(0,num_priors): 
        start_idx = i* 4
        for j in range(0,4):
            prior_variances[i][j]=prior_variances_data[start_idx + j]

    all_loc_preds_q_16 = torch.zeros([num_pred_per_class,num_classes_,4])
    all_conf_scores_q_16 = torch.zeros([num_classes_,num_pred_per_class])
    all_decode_bboxes_q_16=torch.zeros([num_pred_per_class,num_classes_,4])
    if self.quantized==True:
        clamp_min=self.params["qmin_scale"]
        clamp_max=self.params["qmax_scale"]
        scales = self.params["scale_value"]
        shifts = self.params["shift_value"]
        m1_scale = scales[0]
        m1_shift = shifts[0]
        m2_scale = scales[1]
        m2_shift = shifts[1]
        lut = self.constants['lut'].betensor
        zero_index=self.params["zero_index"]

        all_loc_preds_q_16 = torch.round(all_loc_preds)
        all_conf_scores_q_16 = torch.round(all_conf_scores/2)
        prior_bboxes_q_16 = torch.round(prior_bboxes)
        prior_variances_q_16 = torch.round(prior_variances)
        label = 1
      
        for i in range(0,num_priors):
            prior_width_q_16 = prior_bboxes_q_16[i][2] - prior_bboxes_q_16[i][0]#xmax-xmin
            prior_height_q_16 = prior_bboxes_q_16[i][3] - prior_bboxes_q_16[i][1]#ymax-ymin
            prior_center_x_q_16 = torch.round((prior_bboxes_q_16[i][0] + prior_bboxes_q_16[i][2]) / 2.)#xmin+xmax
            prior_center_y_q_16 = torch.round((prior_bboxes_q_16[i][1] + prior_bboxes_q_16[i][3]) / 2.)#ymin+ymax

            # do prior quant
            dh=torch.clip(torch.round(prior_variances[i][0] * all_loc_preds[i][0][0] /m1_scale),-32758,32767)#h
            dw=torch.clip(torch.round(prior_variances[i][1] * all_loc_preds[i][0][1]/ m1_scale),-32768,32767)#w
            dh1=torch.clip(torch.round(prior_variances[i][2] * all_loc_preds[i][0][2]/ m1_scale),-32768,32767)#h
            dw1=torch.clip(torch.round(prior_variances[i][3] * all_loc_preds[i][0][3]/ m1_scale),-32768,32767)#w
    
            temp_x_q=torch.round(dh * prior_width_q_16/ m2_scale)
            temp_y_q=torch.round(dw * prior_height_q_16 / m2_scale)
            decode_bbox_center_x_q = torch.clip( torch.round(temp_x_q*31286/27305) + prior_center_x_q_16, -32768, 32767)
            decode_bbox_center_y_q = torch.clip( torch.round(temp_y_q*31286/27305) + prior_center_y_q_16, -32768, 32767)

            dh_high8=int(dh1/256)+zero_index
            dh_low8=dh1-dh_high8*256
            q_dh=lut[dh_high8]+(lut[dh_high8+1]+lut[dh_high8])*dh_low8/256
            q_dh=lut[dh_high8]

            dw_high8=int(dw1/256)+zero_index
            dw_low8=dw1-dw_high8*256
            q_dw=lut[dw_high8]+(lut[dw_high8+1]+lut[dw_high8])*dw_low8/256
            q_dw=lut[dw_high8]

            decode_bbox_height_q = torch.clip(torch.round(prior_height_q_16 * q_dh /m2_scale ), -32768, 32767)
            decode_bbox_width_q = torch.clip(torch.round(prior_width_q_16 * q_dw /m2_scale), -32768, 32767)

            all_decode_bboxes_q_16[i][label][0]=decode_bbox_center_x_q - torch.round(decode_bbox_width_q / 2) #xmin
            all_decode_bboxes_q_16[i][label][1]=decode_bbox_center_y_q - torch.round(decode_bbox_height_q / 2)#ymin
            all_decode_bboxes_q_16[i][label][2]=decode_bbox_center_x_q + torch.round(decode_bbox_width_q / 2) #xmax
            all_decode_bboxes_q_16[i][label][3]=decode_bbox_center_y_q + torch.round(decode_bbox_height_q / 2)#ymax

    else:
        all_decode_bboxes=torch.zeros([num_pred_per_class,num_classes_,4])
        label = 1
        for i in range(0,num_priors):
            prior_width = prior_bboxes[i][2] - prior_bboxes[i][0]#xmax-xmin
            prior_height = prior_bboxes[i][3] - prior_bboxes[i][1]#ymax-ymin
            prior_center_x = (prior_bboxes[i][0] + prior_bboxes[i][2]) / 2.#xmin+xmax
            prior_center_y = (prior_bboxes[i][1] + prior_bboxes[i][3]) / 2.#ymin+ymax
            if variance_encoded_in_target:
                #variance is encoded in target, we simply need to retore the offset predictions.
                decode_bbox_center_x = all_loc_preds[i][label][0] * prior_width + prior_center_x#bbox.xmin==label loc pred
                decode_bbox_center_y = all_loc_preds[i][label][1]* prior_height + prior_center_y#bbox.ymin
                decode_bbox_width = math.exp(all_loc_preds[i][label][2]) * prior_width#bbox.xmax
                decode_bbox_height = math.exp(all_loc_preds[i][label][3]) * prior_height#bbox.ymax
            else:
                #variance is encoded in bbox, we need to scale the offset accordingly.
                decode_bbox_center_x = prior_variances[i][0] * all_loc_preds[i][0][0] * prior_width + prior_center_x
                decode_bbox_center_y = prior_variances[i][1] * all_loc_preds[i][0][1] * prior_height + prior_center_y
                decode_bbox_width = math.exp(prior_variances[i][2] * all_loc_preds[i][0][2]) * prior_width
                decode_bbox_height = math.exp(prior_variances[i][3] * all_loc_preds[i][0][3]) * prior_height

            all_decode_bboxes[i][label][0]=decode_bbox_center_x - decode_bbox_width / 2. #xmin
            all_decode_bboxes[i][label][1]=decode_bbox_center_y - decode_bbox_height / 2.#ymin
            all_decode_bboxes[i][label][2]=decode_bbox_center_x + decode_bbox_width / 2. #xmax
            all_decode_bboxes[i][label][3]=decode_bbox_center_y + decode_bbox_height / 2.#ymax
  
    if self.quantized==True:
        num_kept = 0
        indices=[]
        num_det = 0
        bboxes = all_decode_bboxes_q_16[:,1,:]
        score_face=all_conf_scores_q_16[1]

        indices=ApplyNMSFast(self,bboxes, score_face, confidence_threshold_, nms_threshold, eta,top_k,indices)
        num_det =num_det + len(indices)
        new_indices=[]
        if keep_top_k_ > -1 and num_det > keep_top_k_:
            score_index_pair=[]
            for label,label_indices in indices:
                scores=all_conf_scores_q_16[c]
                for j in range(0,len(label_indices)):
                    idx = label_indices[j]
                    score_index_pairs.append([[scores[idx]], label, idx])  
            #Keep top k results per image.
            score_index_pairs.sort(key=lambda ele:ele[0])
            score_index_pairs=score_index_pairs[:keep_top_k_]
            #Store the new indices.
            for j in range(0,len(score_index_pairs)):
                label = score_index_pairs[j][1][0]
                idx = score_index_pairs[j][1][1]
                new_indices[label]=idx
            new_indices.append(new_indices)
            num_kept += keep_top_k_
        else:
            new_indices.append(indices)
            num_kept += num_det

        top_data=np.zeros([1,1,num_kept,7],dtype=float)
        top_datanew=np.zeros([1,1,80,7],dtype=float)
        count = 0
        for index in indices:
            bboxes = all_decode_bboxes_q_16[index][label]
            for j in range(0,len(indices)):
                idx = indices[j]
                top_data[0][0][j][0] = 0
                top_data[0][0][j][1] = label
                top_data[0][0][j][2] = all_conf_scores_q_16[label][idx]
                top_data[0][0][j][3] = all_decode_bboxes_q_16[idx][1][0]#xmin()
                top_data[0][0][j][4] = all_decode_bboxes_q_16[idx][1][1]#ymin()
                top_data[0][0][j][5] = all_decode_bboxes_q_16[idx][1][2]#xmax()
                top_data[0][0][j][6] = all_decode_bboxes_q_16[idx][1][3]#ymax()

                top_datanew[0][0][j][0] = 0
                top_datanew[0][0][j][1] = label
                top_datanew[0][0][j][2] = all_conf_scores_q_16[label][idx]
                top_datanew[0][0][j][3] = all_decode_bboxes_q_16[idx][1][0]#xmin()
                top_datanew[0][0][j][4] = all_decode_bboxes_q_16[idx][1][1]#ymin()
                top_datanew[0][0][j][5] = all_decode_bboxes_q_16[idx][1][2]#xmax()
                top_datanew[0][0][j][6] = all_decode_bboxes_q_16[idx][1][3]#ymax()
                count=count+1
        self.outputs[0].betensor = top_datanew
        return top_datanew
    else :
        num_kept = 0
        indices=[]
        num_det = 0
        bboxes = all_decode_bboxes[:,1,:]
        score_face=all_conf_scores[1]
        indices=ApplyNMSFast(self,bboxes, score_face, confidence_threshold_, nms_threshold, eta,top_k,indices)
        num_det =num_det + len(indices)
        new_indices=[]
        if keep_top_k_ > -1 and num_det > keep_top_k_:
            score_index_pair=[]
            for label,label_indices in indices:
                scores=all_conf_scores[c]
                for j in range(0,len(label_indices)):
                    idx = label_indices[j]
                    score_index_pairs.append([[scores[idx]], label, idx])  
            #Keep top k results per image.
            score_index_pairs.sort(key=lambda ele:ele[0])
            score_index_pairs=score_index_pairs[:keep_top_k_]
            #Store the new indices.
            for j in range(0,len(score_index_pairs)):
                label = score_index_pairs[j][1][0]
                idx = score_index_pairs[j][1][1]
                new_indices[label]=idx
            new_indices.append(new_indices)
            num_kept += keep_top_k_
        else:
            new_indices.append(indices)
            num_kept += num_det
        top_data=np.zeros([1,1,num_kept,7],dtype=float)
        top_datanew=np.zeros([1,1,80,7],dtype=float)
        for index in indices:
            bboxes = all_decode_bboxes[index][label]
            for j in range(0,len(indices)):
                idx = indices[j]
                top_data[0][0][j][0] = 0
                top_data[0][0][j][1] = label
                top_data[0][0][j][2] = all_conf_scores[label][idx]
                top_data[0][0][j][3] = all_decode_bboxes[idx][1][0]#xmin()
                top_data[0][0][j][4] = all_decode_bboxes[idx][1][1]#ymin()
                top_data[0][0][j][5] = all_decode_bboxes[idx][1][2]#xmax()
                top_data[0][0][j][6] = all_decode_bboxes[idx][1][3]#ymax()

                top_datanew[0][0][j][0] = 0
                top_datanew[0][0][j][1] = label
                top_datanew[0][0][j][2] = all_conf_scores[label][idx]
                top_datanew[0][0][j][3] = all_decode_bboxes[idx][1][0]#xmin()
                top_datanew[0][0][j][4] = all_decode_bboxes[idx][1][1]#ymin()
                top_datanew[0][0][j][5] = all_decode_bboxes[idx][1][2]#xmax()
                top_datanew[0][0][j][6] = all_decode_bboxes[idx][1][3]#ymax()

        self.outputs[0].betensor = top_datanew
        return top_datanew
        

def ApplyNMSFast(self,bboxes, scores, confidence_threshold_, nms_threshold_, eta_,top_k_,indices):

    if self.quantized==True:
        score_index_vec=[]
        confidence_threshold_q=self.params['nms_confidence']
        for i in range(0,len(scores)):
                if scores[i] > confidence_threshold_q:
                    score_index_vec.append([scores[i], i])

        score_index_vec.sort(reverse = True,key=lambda ele:ele[0])
        if (top_k_ > -1 and top_k_ < len(score_index_vec)):
            score_index_vec=score_index_vec[:top_k_]
        nms_threshold_q=self.params['nms_threshold']
        adaptive_threshold = nms_threshold_q
        indices=[]
        while (len(score_index_vec) != 0):
            idx = score_index_vec[0][1]
            keep = True
            for k in range(0,len(indices)):
                if keep==True:
                    kept_idx = indices[k]
                    overlap_q = JaccardOverlap(self,bboxes[idx], bboxes[kept_idx])
                    overlap=overlap_q[0]
                    if overlap <= adaptive_threshold:
                        keep=True
                    else:
                        keep=False    
                else:
                    break
            if keep==True:
                indices.append(idx)
            del score_index_vec[0]
        return indices
    else:
        #Get top_k scores (with corresponding indices).
        score_index_vec=[]
        #Generate index score pairs.
        for i in range(0,len(scores)):
                if scores[i] > confidence_threshold_:
                    score_index_vec.append([scores[i], i])
        # Sort the score pair according to the scores in descending order
        score_index_vec.sort(reverse = True,key=lambda ele:ele[0])
        #Keep top_k scores if needed.
        if (top_k_ > -1 and top_k_ < len(score_index_vec)):
            score_index_vec=score_index_vec[:top_k_]
        # Do nms.
        adaptive_threshold = nms_threshold_
        indices=[]
        while (len(score_index_vec) != 0):
            idx = score_index_vec[0][1]
            keep = True
            for k in range(0,len(indices)):
                if keep==True:
                    kept_idx = indices[k]
                    overlap_ = JaccardOverlap(self,bboxes[idx], bboxes[kept_idx])
                    overlap=overlap_[0]
                    if overlap <= adaptive_threshold :
                        keep=True
                    else:
                        keep=False    
                else:
                    break
            if keep==True:
                indices.append(idx)
            del score_index_vec[0]
        return indices

def JaccardOverlap(self,bbox1,bbox2):
    if self.quantized:
        normalized=1
        intersect_bbox=[0]*4   
        scales = self.params["scale_value"]
        shifts = self.params["shift_value"]
        m3_scale = scales[2]
        m3_shift = shifts[2]
        m4_scale = scales[3]
        m4_shift = shifts[3]
        if bbox1[3]<bbox1[1] or bbox1[2]<bbox1[0] or bbox2[3]<bbox2[1] or bbox2[2]<bbox2[0]:
            return [500000]
        if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1] :
            return [500000]
        intersect_bbox[0]=max(bbox1[0], bbox2[0])
        intersect_bbox[1]=max(bbox1[1], bbox2[1])
        intersect_bbox[2]=min(bbox1[2], bbox2[2])
        intersect_bbox[3]=min(bbox1[3], bbox2[3])

        intersect_width = intersect_bbox[2] - intersect_bbox[0]
        intersect_height = intersect_bbox[3]- intersect_bbox[1]
        if intersect_width<0 or intersect_height<0:
            return [500000]
        if (intersect_width > 0 and intersect_height > 0) :
            intersect_sizeq = torch.round(intersect_width * intersect_height/ m3_scale)#h
            bbox1_size_ = BBoxSize(self,bbox1,1)
            if(bbox1_size_==0) :return [500000.]
            bbox1_sizeq =bbox1_size_[0]
            bbox2_size_ = BBoxSize(self,bbox2,1)
            if(bbox2_size_==0) :return [500000.]
            bbox2_sizeq=bbox2_size_[0]
            tempq=bbox1_sizeq + bbox2_sizeq - intersect_sizeq
            resq = torch.round(intersect_sizeq / tempq * m4_scale )#h
            return [resq]
        else :
            return [0.]
    else:
        normalized=1
        intersect_bbox=[0]*4   
        if bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1] :
            intersect_bbox=[0]*4 
        else :
            intersect_bbox[0]=max(bbox1[0], bbox2[0])
            intersect_bbox[1]=max(bbox1[1], bbox2[1])
            intersect_bbox[2]=min(bbox1[2], bbox2[2])
            intersect_bbox[3]=min(bbox1[3], bbox2[3])

        if normalized:
            intersect_width = intersect_bbox[2] - intersect_bbox[0]
            intersect_height = intersect_bbox[3]- intersect_bbox[1]
        else :
            intersect_width = intersect_bbox[2] - intersect_bbox[0] + 1
            intersect_height = intersect_bbox[3] - intersect_bbox[1] + 1

        if (intersect_width > 0 and intersect_height > 0) :
            intersect_size = intersect_width * intersect_height
            bbox1_size_ = BBoxSize(self,bbox1,1)
            bbox1_size =bbox1_size_[0]
            bbox2_size_ = BBoxSize(self,bbox2,1)
            bbox2_size=bbox2_size_[0]
            res=intersect_size / (bbox1_size + bbox2_size - intersect_size)
            return [res]
        else :
            return [0.]

def BBoxSize(self,bbox,normalized):
    if self.quantized:
        scales = self.params["scale_value"]
        shifts = self.params["shift_value"]
        m3_scale = scales[2]
        m3_shift = shifts[2]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        res_q = torch.round(width * height/ m3_scale)#h
        return [res_q]

    else:
        if (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
            #If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
            return 0
        else:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if normalized :
                return [width * height]
            else :
                #If bbox is not within range [0, 1].
                return (width + 1) * (height + 1)


@quant_register(OpType.DetectionOutput,2)
def DetectionOutput_quantize(self, *args):
    inp0 = self.inputs[0]
    inp1 = self.inputs[1]
    inp2 = self.inputs[2]
    q_mode_activation = self.attrs["q_mode_activation"]

    loc_data_scale= 1/inp0.scale
    conf_data_scale= 1/inp1.scale
    prior_data_variance_scale= 1/inp2.scale


    out = self.outputs[0]
    out.dtype = inp0.dtype
    out.scale = loc_data_scale
    out.zerop = 0
    out.qbits = 16
    out.qinvariant = False

    #Multiplication quantization parameter
    #s3 max 0.5817 min -0.3661
    scale_s1_1=prior_data_variance_scale
    scale_s2_1=loc_data_scale
    scale_s3_1=0.5817/(2**15-1)
    m_1=scale_s1_1*scale_s2_1/scale_s3_1
    m_scale_1=1/m_1
    m_shift_1=0

    #s3*inp3
    scale_s1_2=(0.5817)/(2**15-1)
    scale_s2_2=loc_data_scale
    scale_s3_2=1.2/(2**15-1)
    m_2=scale_s1_2*scale_s2_2/scale_s3_2
    m_scale_2=1/m_2
    m_shift_2=0

    ####about nms
    scale_s1_3=loc_data_scale
    scale_s2_3=loc_data_scale
    scale_s3_3=1.4/(2**15-1)
    m_3=scale_s1_3*scale_s2_3/scale_s3_3
    m_scale_3=1/m_3
    m_shift_3=0

    scale_s1_4=loc_data_scale
    scale_s2_4=loc_data_scale
    scale_s3_4=1/(2**15-1)
    m_4=scale_s1_4/(scale_s2_4*scale_s3_4)
    m_scale_4=m_4
    m_shift_4=0

    self.params["scale_value"] = [int(m_scale_1), int(m_scale_2),int(m_scale_3),int(m_scale_4)]
    self.params["scale_type"] = Dtype.INT16
    self.params["shift_value"] = [int(m_shift_1), int(m_shift_2),int(m_shift_3),int(m_shift_4)]

    #about exp lut 

    #max 0.5817 min -0.3661
    exp_min=-1
    exp_max=2
    s=np.linspace(exp_min,exp_max,256)
    zero_index=index=np.argwhere(s == 0)
    value=np.exp(s)
    vmax=value.max()
    exp_scale=(256*256-1)/vmax
    exp_q_value=int(m_scale_2)*value
    lut=exp_q_value.astype(np.uint16)

    self.constants["lut"] = Tensor(self.name+"/detection_lut", lut.astype(np.int16))
    self.params["zero_index"]=int(zero_index)
    self.params["qmin_scale"]=int(-2**15-1)
    self.params["qmax_scale"]=int(2**15)

    # about nms
    confidence_threshold_=0.05*int(2**15)#1700
    nms_threshold=0.42*int(2**15)
    self.params['nms_confidence']=confidence_threshold_
    self.params['nms_threshold']=nms_threshold

'''
#Quantization auxiliary function 

def linear_requantize(x, multiplier, shift_bits, zero_point, clamp_min, clamp_max):
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = torch.clamp(torch.round(x_t * (multiplier / (2 ** shift_bits))) - zero_point, clamp_min, clamp_max)
    return y


def linear_quantize_clip(x, scale, zero_point, clamp_min, clamp_max):
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = torch.clamp(torch.round(scale * x_t - zero_point), clamp_min, clamp_max)
    return y

def get_scale_approximation_params(fp32_scale_value, mult_bits, limit=False, mult_bits_ceil=15):
    fp32_scale = fp32_scale_value if isinstance(fp32_scale_value, torch.Tensor) else torch.tensor(fp32_scale_value, dtype=torch.float32)
    mbits = mult_bits if isinstance(mult_bits, torch.Tensor) else torch.tensor(mult_bits)
    #AIFF max support multiplier value is 32767, so we limit mbits to less equal than 15
    mbits = torch.minimum(mult_bits_ceil*torch.ones_like(mbits), mbits)
    shift_bits = torch.log2((2 ** mbits - 1) / fp32_scale).floor()
    if limit:
        shift_bits = torch.minimum(mbits, shift_bits)
    multiplier = (fp32_scale * (2 ** shift_bits)).floor()
    if fp32_scale.dim() < 1 :
        return multiplier.item(), shift_bits.item()
    else :
        return multiplier, shift_bits

def get_linear_quant_params_from_tensor(x, quant_mode, bits, is_signed) :
    assert QuantMode.is_valid(quant_mode), "%s is not one of '%s'" % (quant_mode, str(QuantMode.mode_names()))
    assert bits > 0
    QUANTIZE_ZERO_BAND = torch.finfo(torch.float32).eps

    q_max, q_min = 2 ** bits - 1, 0
    if is_signed :
        if QuantMode.is_full_range(quant_mode) :
            q_max = 2 ** (bits - 1) - 1
            q_min = -1 * q_max - 1
        else :
            q_max = 2 ** (bits - 1) - 1
            q_min = -1 * q_max
    q_range = q_max - q_min

    f_ranges  = torch.tensor(x.max - x.min, dtype=torch.float32, device=x.betensor.device)
    f_zfactor = torch.tensor(x.min, dtype=torch.float32, device=x.betensor.device)
    f_zoffset = torch.zeros_like(f_zfactor)
    if QuantMode.is_per_channel(quant_mode) :
        if QuantMode.is_asymmetric(quant_mode) :
            f_ranges = x.max_key_axis - x.min_key_axis
            f_zfactor = x.min_key_axis
            f_zoffset = q_min * torch.ones_like(f_zfactor)
        else :
            f_ranges = torch.max(x.max_key_axis.abs(), x.min_key_axis.abs())
            if is_signed :
                f_ranges  = f_ranges * 2.0
            f_zfactor = torch.zeros_like(x.min_key_axis)
            f_zoffset = torch.zeros_like(f_zfactor)
    else :
        if QuantMode.is_asymmetric(quant_mode) :
            f_ranges  = torch.tensor(x.max - x.min, dtype=torch.float32, device=x.betensor.device)
            f_zfactor = torch.tensor(x.min, dtype=torch.float32, device=x.betensor.device)
            f_zoffset = q_min * torch.ones_like(f_zfactor)
        else :
            f_ranges = torch.tensor(max(abs(x.max), abs(x.min)), dtype=torch.float32, device=x.betensor.device)
            if is_signed :
                f_ranges = f_ranges * 2.0
            f_zfactor = torch.tensor(0.0, dtype=torch.float32, device=x.betensor.device)
            f_zoffset = torch.zeros_like(f_zfactor)
    q_ranges = q_range * torch.ones_like(f_ranges)
    f_ranges = torch.where(f_ranges < QUANTIZE_ZERO_BAND, q_ranges, f_ranges)
    scale = q_ranges / f_ranges
    zerop = scale.mul(f_zfactor).round() - f_zoffset

    if scale.dim() < 1 :
        return scale.item(), zerop.item(), q_min, q_max
    else :
        return scale, zerop, q_min, q_max

def linear_dequantize(input, scale, zero_point):
    return (input + zero_point) / scale

def cosine_distance(a, b):
    cos = torch.nn.CosineSimilarity(dim=0)
    x = torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a.float()
    y = torch.tensor(b, dtype=torch.float32) if not isinstance(b, torch.Tensor) else b.float()
    ret = cos(x.flatten(), y.flatten()).item()
    return ret
'''