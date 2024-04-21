# 书生大语言模型实战培训营第四次课程笔记

## 课程概述
- **讲师**: 李剑锋、汪周谦、王群，XTuner 贡献者
- - **内容**:
  - 微调理论讲解及 XTuner 介绍
  - XTuner 微调小助手个人认知实战
  - XTuner 微调 llava 图片理解多模态模型实战
    
## 微调理论讲解及 XTuner 介绍
- **微调** (Finetune) 是对预训练大模型进行特定任务或领域的调整，以提高模型在特定任务上的表现。

两种微调范式：

<img width="839" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/8778246a-a001-4551-8970-5296fed92d96">

准备流程：

<img width="884" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/c7c7d389-f0ac-4865-bc73-8a34f70ff0d0">

什么事对话模版？

<img width="889" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/5ea5bf6b-7c56-4d97-8487-2d5c2ac2fcd8">

两种PEFT方法：

<img width="838" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/8ae64e8e-f9eb-406d-a581-c52d99b2b150">

<img width="953" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/64b154ac-c265-43c4-a7a7-59116f6d45a5">


  
- **XTuner** 是一个低门槛的微调框架，支持多种生态适配，具有集成化微调工具，能够在较低显存配置下运行。

<img width="871" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/1cfb68ba-22d2-44f8-8ebb-8629fcdb97fb">

<img width="883" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/03cf6aec-3512-4813-a3b3-c098d3b96541">

<img width="835" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/7c1acc1b-116e-42f5-b692-537702e0d36d">

<img width="875" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/efc03909-4a97-455b-ab03-52acd2350502">


## XTuner 微调小助手个人认知实战
- **实战目标**: 使用XTuner微调一个小型的个人认知助手模型。
- **数据准备**: 通过脚本生成包含输入和输出对的JSON格式数据集。
- **模型选择**: 使用InternLM2-Chat-1.8B作为基础模型进行微调。
- **配置文件**: 选择与任务匹配的配置文件，并根据需要进行修改。
- **训练过程**: 使用XTuner进行模型训练，并通过DeepSpeed技术加速训练过程。

## XTuner 微调 llava 图片理解多模态模型实战
- **多模态LLM**: 结合文本和图像信息，提升模型对图片内容的理解能力。

<img width="799" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/ace34869-43d4-4a0e-a019-aa5bc1ed3b16">


- **LLaVA方案**: 通过训练Image Projector，使LLM具备视觉能力，能够对图片进行描述和理解。

<img width="791" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/a8d8c2c3-6b58-4b45-adba-83ceaefe1cd6">

<img width="877" alt="image" src="https://github.com/soloxue/Intern.LLM.training.camp.S2.2024.03/assets/36829965/4c09711f-38e5-4a02-b034-d805145b6b65">



## 总结
本次课程深入讲解了微调理论和XTuner框架的使用，并通过两个实战案例展示了如何微调大模型以适应特定任务。通过学习，我们了解了微调的基本概念、XTuner框架的特点和使用方法，以及如何构建和训练多模态模型。这些知识和技能对于提升大模型在实际应用中的表现至关重要。
