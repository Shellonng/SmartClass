<template>
  <div class="student-detail">
    <h1>学生详情</h1>
    <p>学生ID: {{ studentId }}</p>
  </div>
</template>

<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
const studentId = props.id
</script>

<style scoped>
.student-detail {
  padding: 24px;
}
</style> 