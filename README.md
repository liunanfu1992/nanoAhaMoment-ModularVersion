This is my replication repository for learning nanoAhaMoment, which maintains the characteristics of single card training but modularizes the project.

You can find their source code on https://github.com/McGill-NLP/nano-aha-moment, thank them for their efforts.

 **Start:**
 
    ```
    PYTHONPATH=$(pwd) deepspeed --num_gpus=1 src/training/trainer.py
    ```