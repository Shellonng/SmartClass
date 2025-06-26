<template>
  <div class="question-detail">
    <h1>题目详情</h1>
    <p>题目ID: {{ questionId }}</p>
  </div>
</template>

<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
const questionId = props.id
</script>

<style scoped>
.question-detail {
  padding: 24px;
}
</style> 