import os
import numpy as np
import sys
# sys.path.append('./backbones/ms-tcn')

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def load_meta(data_root, model_root, result_root, record_root, dataset, split, model_name):
    mapping_file = os.path.join('./dataset/', dataset, 'mapping.txt')

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_actions = len(actions_dict)
    gt_path = os.path.join(data_root, dataset, 'groundTruth/')
    features_path =  os.path.join(data_root, dataset, 'features/')
    vid_list_file = os.path.join(data_root, dataset, 'splits', 'train.split'+str(split)+'.bundle')
    vid_list_file_tst = os.path.join(data_root, dataset, 'splits', 'test.split'+str(split)+'.bundle')

    sample_rate = 1
    if dataset == "50salads":
        sample_rate = 2      
        
    model_dir = os.path.join(model_root, model_name, dataset, 'split_'+str(split))
    result_dir = os.path.join(result_root, model_name, dataset, 'split_'+str(split))
    record_dir = os.path.join(record_root, model_name, dataset)
   
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    print('Created :'+model_dir)
    print('Created :'+result_dir)
    print('Created :'+record_dir)
        
    return actions_dict, num_actions, gt_path, features_path, vid_list_file, vid_list_file_tst, sample_rate, model_dir, result_dir, record_dir

def load_meta_best_eval(data_root, result_root, dataset, split, model_name):
    mapping_file = os.path.join('./dataset/', dataset, 'mapping.txt')

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_actions = len(actions_dict)
    features_path =  os.path.join(data_root, dataset, 'features/')
    vid_list_file_tst = os.path.join(data_root, dataset, 'splits', 'test.split'+str(split)+'.bundle')

    sample_rate = 1
    if dataset == "50salads":
        sample_rate = 2      
    
    result_dir = os.path.join(result_root, model_name, dataset, 'split_'+str(split))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    return actions_dict, num_actions, features_path, vid_list_file_tst, sample_rate, result_dir


def eval_txts(data_root, result_dir, dataset, split, name):
    ground_truth_path = os.path.join(data_root, dataset, 'groundTruth/') 
    recog_path = os.path.join(result_dir)
    file_list = os.path.join(data_root, dataset, 'splits', 'test.split'+str(split)+'.bundle')

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    results = dict()
    
    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = os.path.join(recog_path, vid.split('.')[0])
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            
    accu = 100*float(correct)/total
    edit = (1.0*edit)/len(list_of_videos)

    print("Acc: %.4f" % (accu))
    print('Edit: %.4f' % (edit))
    
    results['accu']=accu
    results['edit']=edit
    
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        results['F1@%0.2f' % (overlap[s])] = f1
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
    return results
   