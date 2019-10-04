export WordTree_DIR=/home/zeyuzhang/PycharmProjects/Hotpot_BERT/expl-tablestore-export-2019-09-10-165215
export Output_DIR=/home/zeyuzhang/PycharmProjects/Hotpot_BERT/output
export TASK_NAME=EPRG

python run_next_explanation.py \
	  --model_type bert \
	    --model_name_or_path bert-base-cased \
	      --task_name $TASK_NAME \
	        --do_train \
		  --do_eval \
		    --do_lower_case \
		      --data_dir $WordTree_DIR \
		        --max_seq_length 128 \
			  --per_gpu_eval_batch_size 32 \
			    --learning_rate 2e-5 \
			      --num_train_epochs 3.0 \
			       --fine_tune_input_dir $Output_DIR/fine_tune/ \
			        --output_dir $Output_DIR/$TASK_NAME/
