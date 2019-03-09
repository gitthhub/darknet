#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;
#ifdef GPU
    l.forward_gpu = forward_detection_layer_gpu;
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

void forward_detection_layer(const detection_layer l, network net)
{
    // // size就是论文中的7, 这里locations表示共有49个格点
    int locations = l.side*l.side;
    int i,j;
    // 输入图像的一部分字节复制到输出？ 输出不应该是由网络前向传播得到的吗？？
    // outputs: 7*7*(3*5+20)
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    // yolov1.cfg中配置为0，没有使用softmax层
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    // 仅在训练阶段才需要计算loss function
    if(net.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        // 也即，delta的shape是：7*7*(3*5+20) * 64
        memset(l.delta, 0, size * sizeof(float));
        // batch = 64  计算一个batch的图像的平均loss
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            // locations = 7*7
            for (i = 0; i < locations; ++i) {
                // truth_index：真实box坐标的索引值
                // coords：包括x, y, w, h
                // 每张图像都有49个grid cell，(b*locations + i)索引到第b张图像的第i个cell
                // (1+l.coords+l.classes)即：5+20
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                // truth中一个box信息(一个grid cell)的存储顺序： confidence classes x y w h
                int is_obj = net.truth[truth_index];
                // 计算置信度confidence的损失
                // l.n=3为每个Grid Cell预测的框的个数
                for (j = 0; j < l.n; ++j) {
                    // p_index:confidence的索引
                    // 对某一张图像，output存储相关预测参数的格式为：
                    // classes(格点总数49*类别总数20)+每个格点对应的三个box的confidence(box1 box2 box3)+每个格点对应的三个box的坐标
                    // 一个格点处的三个box实际也只能预测一个类别，因为此处一个格点只有一个one-hot向量
                    int p_index = index + locations*l.classes + i*l.n + j;
                    // 0表示先假设这些框中都没有物体，计算当前图片当前格点所有box的confidence的平方和
                    // 这里delta信息的存储很重要
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                // 若当前图像的当前格点在ground truth中实际并没有包含物体，则继续下一个格点的计算
                // 结合该函数结尾处loss的计算方式，当一个box实际并不包含物体时，只有其confidence误差被算入loss
                if (!is_obj){
                    continue;
                }

                // 计算类别C的损失
                // 当前图像当前格点的one-hot向量与预测值的delta值
                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }

                // 计算位置信息的损失
                // 提取当前图像当前格点所对应的框的真实坐标  net.truth就是加载数据时的y
                // x y 存储的是0-1之间的数 ？？！
                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                truth.x /= l.side;
                truth.y /= l.side;

                // 遍历当前图像当前格点所预测的三个框，通过比较得出最后的预测框
                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    // 计算预测的框和真实框的iou
                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    // 计算均方根误差(root mean square error)
                    float rmse = box_rmse(out, truth);
                    // 选出iou最大或均方根误差最小的那个框作为最后的预测框
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                // 强制确定一个最后的预测框
                // best_indes: 0 1 2 分别对应box1-box3
                // forced=0
                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }

                // 随机确定一个最后的预测框
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                // 预测框的索引
                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                // 真实框的索引
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                // 计算预测框和真实框的iou  其实就是在选box时的best_iou
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                // 被选中的box的索引
                int p_index = index + locations*l.classes + i*l.n + best_index;
                // 前面假设所有框中都没有物体，并计算confidence损失
                // 这里再把有物体的减掉，并加上正确的confidence损失， 同时更新delta
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                // yolov1.cfg文件中，rescore=1
                // 这里相当于计算confidence误差时，不再拿1减，而是拿预测box与真实box的iou代替1去减
                // output相当于对当前box可靠性的评分，当iou很小且output也很小时，说明此时该框很不可靠
                // 此时使用iou减可以减小对整体loss的影响
                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                // 计算坐标的差
                l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                // 论文中进行了开方，减少box大小不同对loss的不均匀影响
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                // 把iou作为损失，包含x, y, w, h四个参数
                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }
        }

        // 论文中没有用到
        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }

        // 前面 l.cost 的计算其实并没有用，这里才真正计算loss，使用的是前面存储的delta信息
        // mag_array: 求平方和再开方
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}

void backward_detection_layer(const detection_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                dets[index].prob[j] = (prob > thresh) ? prob : 0;
            }
        }
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network net)
{
    if(!net.train){
        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
        return;
    }

    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    forward_detection_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void backward_detection_layer_gpu(detection_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
