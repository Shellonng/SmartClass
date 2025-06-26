<template>
  <div class="class-detail">
    <h1>班级详情</h1>
    <p>班级ID: {{ classId }}</p>
  </div>
</template>

<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
const classId = props.id
</script>

<style scoped>
.class-detail {
  padding: 24px;
}
</style> 