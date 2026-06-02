# PROJECT CONTEXT

## 1. 项目类型

实时 AI 语音转换平台 / 桌面与 Web 客户端配套项目。项目以本地或网络服务方式接收音频流，加载语音转换模型，并返回转换后的音频。

## 2. 技术栈（推测）

- 后端：Python, FastAPI, Uvicorn, python-socketio, PyTorch, ONNX Runtime, NumPy, torchaudio, sounddevice
- 前端：TypeScript, React, Webpack, socket.io-client, Web Audio / AudioWorklet, onnxruntime-web, TensorFlow.js
- 存储：本地文件系统（模型目录、上传目录、静态资源、设置文件）
- 中间件：Socket.IO / ASGI, REST API, Docker

## 3. 项目结构（高层）

```text
.
├── server/
│   ├── MMVCServerSIO.py
│   ├── restapi/
│   ├── sio/
│   ├── voice_changer/
│   ├── downloader/
│   └── data/
├── client/
│   ├── lib/
│   ├── demo/
│   └── python/
├── recorder/
│   ├── src/
│   └── public/
├── docs/
│   ├── index.html
│   └── assets/
├── docker/
├── docker_vcclient/
├── docker_trainer/
├── docker_folder/
├── trainer/
├── script/
├── tutorials/
└── README*.md
```

## 4. 核心模块划分

- `server/MMVCServerSIO.py`：后端启动入口，解析参数、下载必要权重/样例、初始化 REST 与 Socket.IO 应用并启动 Uvicorn。
- `server/restapi/`：FastAPI REST 接口层，提供健康检查、模型/设置操作、文件上传与静态资源挂载。
- `server/sio/`：Socket.IO 实时通信层，接收客户端音频二进制数据并返回转换后的音频。
- `server/voice_changer/`：核心语音转换域模块，管理模型槽位、设备音频、推理器和不同模型类型的加载/转换逻辑。
- `server/downloader/`：启动或加载模型时使用的权重、样例模型下载逻辑。
- `client/lib/`：可复用的 TypeScript 语音转换客户端库，封装前端音频处理和服务通信能力。
- `client/demo/`：React/Webpack 示例或主 GUI 前端，依赖客户端库并支持 Web 版本构建。
- `recorder/`：录音相关前端应用，用于音频录制、波形处理或数据准备。
- `docs/`：构建后的前端静态文件目录，由后端挂载到 `/front`、`/trainer`、`/recorder` 等路径。
- `docker*` 与 `script/`：运行、打包、镜像构建和发布相关工程化脚本。
- `trainer/`：训练数据、配置和中间产物目录，配合训练镜像或外部训练流程使用。

## 5. 主要业务流程

1. 用户启动 `server/MMVCServerSIO.py`，服务解析端口、HTTPS、模型目录和预训练权重路径等参数。
2. 服务初始化全局语音转换参数，必要时下载权重和样例模型，然后创建 `VoiceChangerManager`。
3. `VoiceChangerManager` 扫描/维护模型槽位、GPU 信息和本地音频设备，并按用户选择加载具体模型实现（如 RVC、Beatrice、MMVC、so-vits-svc 等）。
4. 前端 GUI 或客户端库通过 REST API 上传模型、读取设置、切换模型；静态前端资源由 FastAPI 直接挂载提供。
5. 实时转换时，客户端通过 Socket.IO `/test` 命名空间发送 PCM int16 音频块，后端调用 `VoiceChangerManager.changeVoice()` 执行推理。
6. 后端将转换后的音频块和性能指标返回客户端，客户端播放或继续后续音频处理。

## 6. 系统角色关系

- API Server：FastAPI 应用，负责静态资源、REST 控制面、文件上传和模型/设置管理。
- Realtime Gateway：Socket.IO ASGI 服务，负责低延迟音频请求和响应。
- VoiceChangerManager：后端核心协调者，连接模型槽位、设备音频、设置持久化和具体推理器。
- Model Implementations：RVC、Beatrice、MMVC、SoVitsSvc40、DDSP-SVC、DiffusionSVC、LLVC、EasyVC 等具体模型适配层。
- Web Client：React/TypeScript GUI，负责用户交互、音频采集、参数调整和服务通信。
- Client Library：可复用 JS 客户端包，封装 Socket.IO、AudioWorklet、缓存和音频处理逻辑。
- Local Device Worker：服务端本地音频设备处理模块，用于服务器侧输入/输出/监听设备模式。
- Docker / Trainer：容器化运行、训练或客户端打包的辅助角色。

## 7. 关键入口文件

- `server/MMVCServerSIO.py`：Python 后端主入口。
- `server/restapi/MMVC_Rest.py`：FastAPI 应用和路由聚合入口。
- `server/sio/MMVC_SocketIOServer.py`：Socket.IO 服务创建入口。
- `server/sio/MMVC_Namespace.py`：实时音频消息处理入口。
- `server/voice_changer/VoiceChangerManager.py`：语音转换核心协调入口。
- `server/voice_changer/ModelSlotManager.py`：模型槽位管理入口。
- `client/lib/src/`：JS/TS 客户端库源码入口目录。
- `client/demo/src/`：前端 GUI 源码入口目录。
- `recorder/src/index.tsx`：录音前端入口。
- `start_docker.sh`、`start2.sh`、`start_v0.1.sh`：项目启动脚本。
- `docker/Dockerfile`、`docker_vcclient/Dockerfile`、`docker_trainer/Dockerfile`：容器构建入口。

## 8. 总体理解（总结）

这是一个面向实时语音转换的多端项目，核心是 Python 后端中的 `VoiceChangerManager` 和多个模型适配实现。后端同时提供 REST 控制面和 Socket.IO 实时音频通道，前端通过 React/TypeScript GUI 或客户端库采集音频、配置模型并播放转换结果。模型、上传文件、样例和设置主要通过本地目录管理，适合桌面、本地服务、网络服务和 Docker 场景。项目支持多种语音转换模型，架构上通过模型槽位和模型生成器隔离不同模型格式。整体业务主线是“加载模型与参数 → 接收音频块 → 推理转换 → 返回音频块”。
