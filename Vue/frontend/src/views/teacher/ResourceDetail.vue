<template>
  <div class="resource-detail">
    <h1>资源详情</h1>
    <p>资源ID: {{ resourceId }}</p>
  </div>
</template>

<script setup lang="ts">
interface Props {
  id: string
}

const props = defineProps<Props>()
const resourceId = props.id
</script>

<style scoped>
.resource-detail {
  padding: 24px;
}
</style> 