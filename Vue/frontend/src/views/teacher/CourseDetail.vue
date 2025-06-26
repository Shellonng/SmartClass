<template>
  <div class="course-detail">
    <h1>课程详情</h1>
    <p>课程ID: {{ courseId }}</p>
  </div>
</template>

<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
const courseId = props.id
</script>

<style scoped>
.course-detail {
  padding: 24px;
}
</style> 