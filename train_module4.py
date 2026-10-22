"""Train the Module 4 budget model once:   python train_module4.py <path to 3_0_my_model_data_train.jsonl> [out.joblib]"""
import sys, module4
module4.train(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "module4_budget.joblib")
