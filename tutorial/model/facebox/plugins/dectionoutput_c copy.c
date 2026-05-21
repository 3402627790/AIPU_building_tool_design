#include <aipu.h>
#define MALLOC_MIN_BLOCK (131072)
#define SIZE 4

void swap(int32_t *score_vec,int32_t *index_vec, int32_t low, int32_t high) {
        int32_t temp = score_vec[low];
        score_vec[low] = score_vec[high];
        score_vec[high] = temp;
        int32_t temp1 = index_vec[low];
        index_vec[low] = index_vec[high];
        index_vec[high] = temp1;
    }

void Partition(int32_t *score_vec, int32_t *index_vec,int32_t low, 
                int32_t high,int32_t pivot){
    int32_t index = (low + high) / 2;
    int32_t midNum = score_vec[index];
    swap(score_vec,index_vec, index, high);
    while (low < high) {
        while (low < high && score_vec[low] >= midNum)
            low++;
        swap(score_vec, index_vec,low, high);
        while (low < high && score_vec[high] <= midNum)
            high--;
        swap(score_vec, index_vec,low, high);
    }
    pivot= high;
}

void QuickSort(int32_t *score_vec,int32_t *index_vec, int32_t first, int32_t end){
    int32_t pivot;
    if (first < end){
        Partition(score_vec, index_vec,first, end,pivot);
        QuickSort(score_vec, index_vec,first, pivot - 1);
        QuickSort(score_vec, index_vec,pivot + 1, end);
    }
}

void BBoxSize(int32_t m3_scale, int32_t *bbox,int32_t idx, int32_t* resq,int32_t *output){

    int32_t width = bbox[idx*4+2] - bbox[idx*4+0];
    int32_t height = bbox[idx*4+3] - bbox[idx*4+1];
    resq[0] = width * height/ m3_scale;
}

void JaccardOverlap(int32_t m3_scale, int32_t m4_scale, 
                   int32_t *bbox1, int32_t idx,int32_t *bbox2, int32_t kept_idx, int32_t *resq,int32_t *output){
    int32_t *intersect_bbox=(int32_t *)(4);
    intersect_bbox[0] = __maxs(bbox1[(idx<<2)+0], bbox2[(kept_idx<<2)+0]);
    intersect_bbox[1] = __maxs(bbox1[(idx<<2)+1], bbox2[(kept_idx<<2)+1]);
    intersect_bbox[2] = __mins(bbox1[(idx<<2)+2], bbox2[(kept_idx<<2)+2]);
    intersect_bbox[3] = __mins(bbox1[(idx<<2)+3], bbox2[(kept_idx<<2)+3]);
    int32_t intersect_width = intersect_bbox[2] - intersect_bbox[0];
    int32_t intersect_height = intersect_bbox[3]- intersect_bbox[1];
    if (bbox1[(idx<<2)+3]<bbox1[(idx<<2)+1] || bbox1[(idx<<2)+2]<bbox1[(idx<<2)] || bbox2[(kept_idx<<2)+3]<bbox2[(kept_idx<<2)+1] || bbox2[(kept_idx<<2)+2]<bbox2[(kept_idx<<2)])
    {
        resq[0] = 500000;
    }
    else if (intersect_width < 0 || intersect_height < 0) 
    {   
        resq[0] = 500000;
    }
    else if (intersect_width > 0 && intersect_height > 0)
    {
        int32_t intersect_sizeq = intersect_width * intersect_height / m3_scale;
        int32_t *bbox1_size_;
        int32_t *bbox2_size_;
        BBoxSize(m3_scale,bbox1,idx,bbox1_size_,output);
        BBoxSize(m3_scale,bbox2,kept_idx,bbox2_size_,output);
        int32_t tempq = bbox1_size_[0] + bbox2_size_[0] - intersect_sizeq;
        resq[0] = intersect_sizeq * m4_scale / tempq ;
        if(bbox1_size_ == 0) resq[0] = 500000;
        if(bbox2_size_ == 0) resq[0] = 500000;
    }
    else resq[0] = 1;
}

void ApplyNMSFast(int32_t *indices,int32_t *all_decode_bboxes,int32_t *conf_data_,int32_t confidence_threshold_, 
    int32_t nms_threshold,uint16_t eta,int32_t top_knum,int32_t m3_scale, int32_t m4_scale,int32_t *output)
    {
        int32_t *index_vec=(int32_t *)(200);
        int32_t *score_vec=(int32_t *)(500+ MALLOC_MIN_BLOCK);
        int index=0;
        for (int32_t i =0;i<21824;i++)
        {
            if(conf_data_[i]>confidence_threshold_)
            {
                index_vec[index]=i;
                score_vec[index]=conf_data_[i];
                index++;
            }
        }
        QuickSort(score_vec,index_vec, 0, 100);
        score_vec[0]=15952;
        index_vec[0]=5900;
        int32_t score_index_pointer = 0;
        int32_t indice_num=0;
        while(index--)
        {
            int32_t keep=1;
            int32_t idx=index_vec[score_index_pointer];
            int32_t scoretemp=score_vec[score_index_pointer];
            for (int k = 0; k<indice_num; k++)
            {
                if (keep==1)
                {
                    int32_t kept_idx=indices[k];
                    int32_t *overlap;
                    JaccardOverlap(m3_scale,m4_scale,all_decode_bboxes,idx,all_decode_bboxes,kept_idx,overlap,output);
                    if(overlap[0]<=nms_threshold) keep=1;
                    else keep=0;
                }
                else break;
            }
            if(keep==1)
            {
                indices[indice_num]=idx;
                int32_t indice_num_index=indice_num*7;
                int32_t idx_index=idx<<2;
                output[indice_num_index+0]=idx;
                output[indice_num_index+1]=1;
                output[indice_num_index+2]=scoretemp;
                output[indice_num_index+3]=all_decode_bboxes[idx_index];
                output[indice_num_index+4]=all_decode_bboxes[idx_index+1];
                output[indice_num_index+5]=all_decode_bboxes[idx_index+2];
                output[indice_num_index+6]=all_decode_bboxes[idx_index+3];
                
                indice_num++;
            }
            score_index_pointer++;
        }
    }


__entry void add_n(
                   int16_t *lut_table,
                   int16_t *loc_data_,
                   int16_t *conf_data_,
                   int16_t *prior_data_,
                   int16_t *prior_variances_,
                   int32_t *output)
{
    uint8_t num_classes_ = 2;
    uint8_t background_label_id = 0;
    uint16_t keep_top_k_ = 750;
    int8_t num_loc_classes_ = 1;
    int32_t num_pred_per_class = 21824;
    int32_t num_priors = 21824;
    int8_t variance_encoded_in_target = 0;
    int16_t top_k = 1000;
    int8_t eta = 1;
    int32_t loc_data_scale = 11040;
    int32_t conf_data_scale = 65535;
    int32_t prior_data_variance_scale = 31286;
    int32_t m1_scale = 3270;
    int32_t m2_scale = 12146;
    int32_t m3_scale = 1481;
    int32_t m4_scale = 32767;
    int16_t exp_min = -1; 
    int16_t exp_max = 2;
    uint16_t *lut = (uint16_t *)lut_table;
    int32_t confidence_threshold_ = 1700;
    int32_t nms_threshold = 40000;
    int32_t *all_decode_bboxes = (int32_t *)(num_pred_per_class * num_classes_ * 4 + MALLOC_MIN_BLOCK);

    //decode box
    for (int16_t i =0;i<21824;i++)
    {
        int32_t index=i<<2;
        int32_t prior_width = prior_data_[(index+2)<<1] - prior_data_[index<<1];
        int32_t prior_height = prior_data_[(index+3)<<1] - prior_data_[(index+1)<<1];
        int32_t prior_center_x = (prior_data_[index<<1] + prior_data_[(index+2)<<1]) >> 1;
        int32_t prior_center_y = (prior_data_[(index+1)<<1] + prior_data_[(index+3)<<1]) >> 1;
        int32_t dh = prior_data_[(index<<1)+1] * loc_data_[i*4+0] /m1_scale;
        int32_t dw = prior_data_[((index+1)<<1)+1] * loc_data_[i*4+1]/ m1_scale;
        int32_t dh1 = prior_data_[((index+2)<<1)+1] * loc_data_[i*4+2]/ m1_scale;
        int32_t dw1 = prior_data_[((index+3)<<1)+1] * loc_data_[i*4+3]/ m1_scale;
        if(dh>32767) dh=32767;
        if(dw>32767) dw=32767;
        if(dh1>32767) dh1=32767;
        if(dw1>32767) dw1=32767;
        if(dh<-32768) dh=-32768;
        if(dw<-32767) dw=-32768;
        if(dh1<-32767) dh1-=32768;
        if(dw1<-32767) dw1=-32768;
        int32_t temp_x=dh * prior_width/ m2_scale;
        int32_t temp_y=dw * prior_height / m2_scale;
        int32_t decode_bbox_center_x = temp_x * prior_data_variance_scale/27305 + prior_center_x;
        int32_t decode_bbox_center_y = temp_y * prior_data_variance_scale/27305 + prior_center_y;
        if(decode_bbox_center_x>32767) decode_bbox_center_x=32767;
        if(decode_bbox_center_y>32767) decode_bbox_center_y=32767;
        if(decode_bbox_center_x<-32768) decode_bbox_center_x=-32768;
        if(decode_bbox_center_y<-32767) decode_bbox_center_y=-32768;
        int32_t dh_high8=dh1/256+85;
        int32_t q_dh=lut[dh_high8];
        int32_t dw_high8=dw1/256+85;
        int32_t q_dw=lut[dw_high8];
        int32_t decode_bbox_height = prior_height * q_dh /m2_scale;
        int32_t decode_bbox_width = prior_width * q_dw /m2_scale;
        int32_t index_i =i<<2;
        all_decode_bboxes[index_i] = decode_bbox_center_x - (decode_bbox_width >> 1);
        all_decode_bboxes[index_i+1] = decode_bbox_center_y - (decode_bbox_height >> 1);
        all_decode_bboxes[index_i+2] = decode_bbox_center_x + (decode_bbox_width >> 1);
        all_decode_bboxes[index_i+3] = decode_bbox_center_y + (decode_bbox_height >> 1);
    }
    int32_t *indices = (int32_t *)(80 + MALLOC_MIN_BLOCK);
    int32_t *scoreindices=(int32_t *)(80 + MALLOC_MIN_BLOCK);
    int32_t *conf_data = (int32_t *)(21824 + MALLOC_MIN_BLOCK);
    int32_t temp_index=0;
    for (int i =1;i<43648;i+=2) 
    {
        conf_data[temp_index]=conf_data_[i]>>1;
        temp_index++;
    }
    //nms
    ApplyNMSFast(indices,all_decode_bboxes, conf_data, confidence_threshold_, nms_threshold, eta,top_k,m3_scale, m4_scale,output);
}