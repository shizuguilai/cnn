#### 首先要更新源（国内服务器）
https://blog.csdn.net/m0_37818883/article/details/103842840?utm_term=linux%E6%9B%B4%E6%96%B0%E6%BA%90&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-2-103842840&spm=3001.4430

#### 文件比较多可以这样
git config --global core.compression 0
git clone --depth 1 <github项目url>

//cd to your newly created directory//

git fetch --unshallow 
#### 然后进行git常规操作，最后再pull
git pull --all
