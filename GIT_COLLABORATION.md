# Git 团队协作指南

## 📋 目录
1. [初始设置](#初始设置)
2. [日常协作流程](#日常协作流程)
3. [常用命令速查](#常用命令速查)
4. [解决冲突](#解决冲突)
5. [最佳实践](#最佳实践)

---

## 🚀 初始设置

### 步骤 1: 创建团队仓库

**选项 A: 使用 GitHub（推荐）**
1. 一位成员在 GitHub 创建新仓库（如 `Final_Project_Team`）
2. 不要初始化 README（因为本地已有代码）

**选项 B: 使用 Gitee（国内更快）**
1. 在 Gitee 创建新仓库
2. 同样不要初始化 README

### 步骤 2: 所有成员配置 Git

```bash
# 设置用户名和邮箱（每个成员都要设置）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"

# 查看配置
git config --list
```

### 步骤 3: 添加远程仓库

**对于第一个成员（已有代码的成员）：**

```bash
# 先提交当前更改
git add .
git commit -m "Initial commit: project setup"

# 添加新的远程仓库（替换为你的仓库地址）
git remote add team-origin https://github.com/你的用户名/Final_Project_Team.git

# 或者使用 Gitee
# git remote add team-origin https://gitee.com/你的用户名/Final_Project_Team.git

# 推送到新仓库
git push -u team-origin master
```

**对于其他成员（克隆仓库）：**

```bash
# 克隆仓库到本地
git clone https://github.com/你的用户名/Final_Project_Team.git
cd Final_Project_Team

# 或者使用 Gitee
# git clone https://gitee.com/你的用户名/Final_Project_Team.git
```

---

## 🔄 日常协作流程

### 推荐工作流程：分支策略

#### 方案 1: 功能分支（Feature Branch）- 推荐

每个成员在自己的分支上工作：

```bash
# 1. 从主分支创建自己的功能分支
git checkout -b feature/ppo-algorithm    # 成员A：PPO算法
git checkout -b feature/actor-critic      # 成员B：Actor-Critic
git checkout -b feature/integration       # 成员C：整合和对比

# 2. 在自己的分支上开发
# ... 编写代码 ...

# 3. 提交更改
git add .
git commit -m "实现PPO算法的核心逻辑"

# 4. 推送到远程
git push origin feature/ppo-algorithm

# 5. 在 GitHub/Gitee 上创建 Pull Request (PR)
# 或者直接合并到主分支（如果团队小）
```

#### 方案 2: 简单协作（适合3人小团队）

直接在 master 分支工作，但每次工作前先拉取最新代码：

```bash
# 每天开始工作前
git pull origin master

# 完成工作后
git add .
git commit -m "描述你的更改"
git push origin master
```

---

## 📝 常用命令速查

### 基础命令

```bash
# 查看状态
git status

# 查看更改内容
git diff

# 添加文件到暂存区
git add <文件名>           # 添加单个文件
git add .                  # 添加所有更改

# 提交更改
git commit -m "提交信息"

# 查看提交历史
git log
git log --oneline          # 简洁版本

# 推送到远程
git push origin master

# 拉取最新代码
git pull origin master
```

### 分支操作

```bash
# 创建新分支
git checkout -b <分支名>

# 切换分支
git checkout <分支名>

# 查看所有分支
git branch
git branch -a              # 包括远程分支

# 合并分支
git checkout master
git merge <分支名>

# 删除分支
git branch -d <分支名>
```

### 远程操作

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add <名称> <URL>

# 拉取远程更改
git fetch origin
git pull origin master

# 推送本地更改
git push origin master
```

---

## ⚠️ 解决冲突

### 当出现冲突时：

```bash
# 1. 先拉取最新代码
git pull origin master

# 2. 如果出现冲突，Git 会提示哪些文件冲突
# 打开冲突文件，你会看到类似这样的标记：

<<<<<<< HEAD
你的代码
=======
队友的代码
>>>>>>> branch-name

# 3. 手动解决冲突：
#    - 删除冲突标记（<<<<<<<, =======, >>>>>>>）
#    - 保留需要的代码
#    - 确保代码逻辑正确

# 4. 标记冲突已解决
git add <冲突文件>

# 5. 完成合并
git commit -m "解决合并冲突"
```

### 避免冲突的技巧：

1. **频繁拉取更新**：每天开始工作前先 `git pull`
2. **分工明确**：不同成员负责不同文件
3. **及时提交**：完成一个小功能就提交，不要积累太多更改
4. **沟通**：修改共享文件前先和队友沟通

---

## ✅ 最佳实践

### 1. 提交信息规范

```bash
# 好的提交信息
git commit -m "实现PPO算法的策略网络"
git commit -m "修复Actor-Critic的价值函数计算bug"
git commit -m "添加超参数调优实验脚本"

# 不好的提交信息（避免）
git commit -m "更新"
git commit -m "修复"
git commit -m "asdf"
```

### 2. 文件组织

创建 `.gitignore` 文件，忽略不需要版本控制的文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# IDE
.idea/
.vscode/
*.swp
*.swo

# 模型文件（如果太大）
*.h5
*.keras
artifacts/*.keras

# 数据文件
*.csv
*.png
scores/*.csv
scores/*.png

# 系统文件
.DS_Store
Thumbs.db
```

### 3. 工作流程建议

**每天开始工作：**
```bash
git pull origin master    # 拉取最新代码
git status                # 查看状态
```

**完成一个功能后：**
```bash
git add .
git commit -m "清晰的提交信息"
git push origin master
```

**遇到问题：**
```bash
git log                   # 查看历史
git diff                  # 查看更改
git stash                 # 临时保存未提交的更改
git stash pop             # 恢复保存的更改
```

### 4. 团队协作约定

- ✅ **分工明确**：每个成员负责不同的算法文件
- ✅ **及时沟通**：修改共享文件前先讨论
- ✅ **小步提交**：完成一个小功能就提交，不要等全部完成
- ✅ **测试后再推送**：确保代码能运行再推送
- ✅ **代码审查**：重要更改前让队友review

---

## 🆘 常见问题

### Q: 误删了文件怎么办？
```bash
git checkout -- <文件名>    # 恢复文件
```

### Q: 想撤销最后一次提交？
```bash
git reset --soft HEAD~1     # 保留更改，撤销提交
git reset --hard HEAD~1     # 完全撤销（危险！）
```

### Q: 想查看某个文件的修改历史？
```bash
git log <文件名>
git diff HEAD~1 <文件名>    # 查看与上一次提交的差异
```

### Q: 本地有未提交的更改，但需要切换分支？
```bash
git stash                  # 暂存更改
git checkout <其他分支>
git stash pop              # 恢复更改
```

---

## 📚 学习资源

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Gitee 帮助文档](https://gitee.com/help)

---

## 💡 快速开始检查清单

- [ ] 所有成员安装 Git
- [ ] 所有成员配置用户名和邮箱
- [ ] 创建团队仓库（GitHub/Gitee）
- [ ] 第一个成员推送初始代码
- [ ] 其他成员克隆仓库
- [ ] 创建 `.gitignore` 文件
- [ ] 约定分支策略和工作流程
- [ ] 开始协作开发！

---

**祝你们协作顺利！** 🎉

