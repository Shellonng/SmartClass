<template>
  <div class="task-detail">
    <h1>任务详情</h1>
    <p>任务ID: {{ taskId }}</p>
  </div>
</template>

<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
const taskId = props.id
</script>

<style scoped>
.task-detail {
  padding: 24px;
}
</style> 