# 上传到 GitHub 私有仓库 - 操作步骤

## 第一步：在 GitHub 创建私有仓库

1. 打开浏览器，访问：**https://github.com/new**
2. 使用你的 GitHub 账号登录（邮箱：3301088903@qq.com）
3. 填写仓库信息：
   - **Repository name**：`bt`（或你喜欢的名称）
   - **Description**：可选，如 "AIPU Building Tool Design"
   - **选择 Private（私有）** ← 重要！确保只有你能访问
   - **不要**勾选 "Add a README file"（本地已有代码）
4. 点击 **Create repository**

## 第二步：在本地推送代码

创建好仓库后，GitHub 会显示仓库地址。在 PowerShell 中执行：

```powershell
cd C:\Users\33010\Desktop\bt

# 添加远程仓库（将 YOUR_USERNAME 替换为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/bt.git

# 推送代码到 GitHub
git branch -M main
git push -u origin main
```

**注意**：首次推送时，GitHub 会要求你登录验证身份。如果使用 HTTPS，可能需要输入用户名和密码（或 Personal Access Token）。

## 如果使用 SSH（推荐，免密推送）

如果你已配置 SSH 密钥，可以使用：

```powershell
git remote add origin git@github.com:YOUR_USERNAME/bt.git
git branch -M main
git push -u origin main
```

---

✅ 本地已完成：Git 初始化、添加文件、首次提交（97 个文件）
