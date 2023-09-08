# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:19:53 2023

@author: i368o351
"""

ab = res0_all2[0] -filters.threshold_sauvola(res0_all2[0],11);
 _ =plt.figure() ; _=plt.imshow(ab)
 
_ =plt.figure() 
f,ax = plt.subplots(3,5,figsize=(50,50)) 
ax[0][0].imshow(a01, cmap='gray_r')
ax[1][0].imshow(a01, cmap='gray_r')
ax[2][0].imshow(a01, cmap='gray_r')
for idx,elem in enumerate(res0_all2): 
    mod_elem = elem - filters.threshold_sauvola(elem,51); alpha = 0.1
    mod_thr = (1-alpha) * mode(mod_elem.ravel()) + alpha*np.max(mod_elem)
    ax[0][idx+1].hist(elem.ravel()); ax[0][idx+1].axvline(mod_thr,c='red'); ax[0][idx+1].set_title(f'{model_names[idx]}');
    ax[1][idx+1].imshow( elem)
    ax[2][idx+1].imshow( np.where(mod_elem>mod_thr,1,0))
    

min_layer_loc = 20
Nt,Nx = abin.shape
final_return_segment = np.zeros_like(abin)

all_cols = []

for col_idx in range(Nx):    

    repeat_tuple = [ (k,sum(1 for _ in groups)) for k,groups in groupby(abin[:,col_idx]) ]
    # Cumulate the returned index
    rep_locs = np.cumsum([ item[1] for item in repeat_tuple])
    repeat_tuple2 = [(repeat_tuple[idx][0],repeat_tuple[idx][1],rep_locs[idx] )  for idx in range(len(rep_locs))]
    all_cols.append(repeat_tuple2)
    
    one_locs_idx = [ rep_locs[idx] for idx,iter in enumerate(repeat_tuple) if iter[0] ==1 and rep_locs[idx] > min_layer_loc]
    for iter in range(len(one_locs_idx)):
        final_return_segment[ one_locs_idx[iter]: min( one_locs_idx[-1]+20,Nt-1), col_idx ] = iter+1
        
        
ab2 = modal(final_return_segment.astype(np.uint8), rectangle(25,201) ) 


for each_col in range(len(all_cols)):
    for each_row in range(len(each_col)):
        
        anew[ each_row,each_col ] = all_cols[each_col][each_row][1]
        
        
# 
        
        
        
        
        

