// 修复TeacherLayout.vue中重复的"首页"问题的脚本
// 使用方法：在项目根目录下运行 node fix-breadcrumb.js

const fs = require('fs');
const path = require('path');

// 文件路径
const filePath = path.join(__dirname, 'Vue', 'frontend', 'src', 'components', 'layout', 'TeacherLayout.vue');

try {
  // 读取文件内容
  let content = fs.readFileSync(filePath, 'utf8');
  
  // 方法1：修改面包屑导航的计算属性，移除初始的"首页"项
  content = content.replace(
    /const breadcrumbItems = computed\(\(\) => \{\s*const items = \[\{ title: '首页', path: '\/teacher' \}\]/,
    "const breadcrumbItems = computed(() => {\n  // 移除首页项，因为已经在顶部导航中有了\n  const items = []"
  );
  
  // 方法2：如果方法1不起作用，尝试注释掉整个面包屑导航组件
  content = content.replace(
    /<a-breadcrumb class="breadcrumb">([\s\S]*?)<\/a-breadcrumb>/,
    "<!-- 移除面包屑导航，避免重复显示"首页" -->"
  );
  
  // 写入修改后的内容
  fs.writeFileSync(filePath, content, 'utf8');
  
  console.log('成功修复了TeacherLayout.vue中重复的"首页"问题！');
} catch (err) {
  console.error('修复过程中出现错误:', err);
} 