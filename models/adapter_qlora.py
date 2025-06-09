from peft import get_peft_model, LoraConfig, TaskType

def inject_qlora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["classifier"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    return get_peft_model(model, config)