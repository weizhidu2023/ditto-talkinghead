关于ditto推理

# 1. 首先尝试最简单的 pip 安装
pip install nvidia-cudnn-cu12==8.9.7.29

# 2. 设置环境变量
export LD_LIBRARY_PATH=~/miniconda3/envs/ditto/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 3. 测试 TensorRT
python -c "import tensorrt as trt; print('成功')"


#原始解开的yml环境版本号不对（13.0.3）
pip uninstall cuda-python
# 或者安装 11.8.7（推荐，包含更多修复）
pip install cuda-python==11.8.7

pip install matplotlib



# 查看所有本地分支
git branch

# 查看当前分支
git status

#转换到某分支
git checkout main/train

++++++++++++++++++++++++++++++++++++++++++
# 1. 进入你已修改的本地仓库目录
cd 仓库目录

# 2. 查看当前远程仓库配置
git remote -v
# 显示：origin  https://github.com/原作者/仓库名.git

# 3. 重命名原来的远程仓库
git remote rename origin upstream

# 4. 添加你自己的Fork作为新的origin
git remote add origin https://github.com/你的用户名/仓库名.git

# 5. 验证设置
git remote -v
# 应该显示：
# origin    https://github.com/你的用户名/仓库名.git (fetch)
# origin    https://github.com/你的用户名/仓库名.git (push)
# upstream  https://github.com/原作者/仓库名.git (fetch)
# upstream  https://github.com/原作者/仓库名.git (push)

# 6. 创建并切换到新分支（保存你的修改）
git checkout -b my-customizations

# 7. 推送到你的Fork仓库
git push -u origin my-customizations
+++++++++++++++++++++++++++++++++++++++++++++++
# 切换到你的自定义分支
git checkout my-customizations

# 进行修改...
# 编辑文件...

# 提交修改
git add .
git commit -m "描述你的修改"

# 推送到你的仓库
git push origin my-customizations

==============配置train分支环境====================
#pyav变为av并更改渠道
#ffmpeg=4.3版本与av11.0.0版本不匹配，最终改为  - ffmpeg>=4.3.2,<4.4.0a0     - av=8.1.0





=============扰动实验================
python single_frame_perturb.py \
  --image ./example/image.png \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
  --data_root ./checkpoints/ditto_trt_Ampere_Plus \
  --batch_all \
  --batch_delta -0.02 \
  --output_dir ./outputs/perturb_test
  （执行63维的所有扰动）

python single_frame_perturb.py \
  --image ./example/image.png \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
  --data_root ./checkpoints/ditto_trt_Ampere_Plus \
  --dimension 49 \
  --delta -0.01 \
  --output_dir ./outputs/perturb_test_single

