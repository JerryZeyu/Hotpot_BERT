import os
import pandas as pd
import numpy as np
single_question_ranked_score_list=[]
MAP_dict = {}
for path, _, files in os.walk('debug_output/'):
    for file in sorted(files, key=lambda file:int(file.split('_')[1].split('.')[0])):
        df_all = pd.read_csv(os.path.join(path, file), sep='\t')

        # for id, item in enumerate(df_all['Supporting Sentence'].unique()):
        #     df_temp = df_all[df_all['Supporting Sentence']==item].copy()
        df_all.sort_values(by='Score_1', ascending=False, inplace=True)
        list_temp = df_all['IsRowGold'].tolist()
            #print(list_temp)
        ranked_list = [i+1 for i, x in enumerate(list_temp) if x == 1]
            #print(ranked_list)
        ranked_score = []
        for ids, ranked_location in enumerate(ranked_list):
            ranked_score.append((ids+1.0)/(ranked_location*1.0))
        single_question_ranked_score_list.append(np.mean(ranked_score))

        MAP_dict['question_'+file.split('_')[1].split('.')[0]]=np.mean(ranked_score)


final_all_question_ranked_score = np.mean(single_question_ranked_score_list)
with open('./questions_map_eachrow.csv','w+') as file:
    file.write('{}\t{}\n'.format('Query', 'MAP'))
    for item in MAP_dict.keys():
        file.write('{}\t{}\n'.format(str(item),str(MAP_dict[item])))
    file.write('{}\t{}\n'.format('All Questions MAP',str(final_all_question_ranked_score)))

