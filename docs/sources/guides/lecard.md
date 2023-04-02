# LeCaRD类案检索

本文档涵盖了如何在中文预料——[LeCaRD类案检索数据集](https://github.com/myx666/LeCaRD)上训练基本的Dense Retriever（DPR）。



```bash
python run.py DPR base=LECARD ++plm=bert-chinese ++batch_size=4

python run.py CrossEnc base=LECARD ++query_length=256
```
