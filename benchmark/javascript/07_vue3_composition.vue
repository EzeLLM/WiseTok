<template>
  <div class="data-table-container">
    <header class="table-header">
      <h2>{{ title }}</h2>
      <div class="controls">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search records..."
          class="search-input"
        />
        <button @click="addNewRow" class="btn-primary">
          + Add Record
        </button>
      </div>
    </header>

    <div class="filters">
      <label>
        <input v-model="showActive" type="checkbox" />
        Show Active Only
      </label>
      <select v-model="sortBy" @change="handleSort">
        <option value="name">Sort by Name</option>
        <option value="date">Sort by Date</option>
        <option value="value">Sort by Value</option>
      </select>
    </div>

    <table class="data-table" v-if="filteredAndSorted.length">
      <thead>
        <tr>
          <th>Name</th>
          <th>Email</th>
          <th>Status</th>
          <th>Value</th>
          <th>Created</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(item, index) in filteredAndSorted"
          :key="item.id"
          :class="{ 'row-highlight': selectedId === item.id }"
          @click="selectRow(item.id)"
        >
          <td>{{ item.name }}</td>
          <td>{{ item.email }}</td>
          <td>
            <span :class="`badge badge-${item.status}`">
              {{ item.status }}
            </span>
          </td>
          <td>{{ formatCurrency(item.value) }}</td>
          <td>{{ formatDate(item.createdAt) }}</td>
          <td class="actions">
            <button @click.stop="editRow(item)" class="btn-sm">Edit</button>
            <button @click.stop="deleteRow(index)" class="btn-sm btn-danger">Delete</button>
          </td>
        </tr>
      </tbody>
    </table>

    <div v-else class="empty-state">
      <p>No records found</p>
    </div>

    <div class="pagination">
      <button
        @click="previousPage"
        :disabled="currentPage === 1"
        class="btn-page"
      >
        Previous
      </button>
      <span class="page-info">Page {{ currentPage }} of {{ totalPages }}</span>
      <button
        @click="nextPage"
        :disabled="currentPage === totalPages"
        class="btn-page"
      >
        Next
      </button>
    </div>

    <modal v-if="showEditModal" @close="closeModal">
      <template #header>
        <h3>{{ editingItem ? 'Edit Record' : 'New Record' }}</h3>
      </template>
      <template #body>
        <form @submit.prevent="saveChanges">
          <div class="form-group">
            <label>Name:</label>
            <input v-model="formData.name" type="text" required />
          </div>
          <div class="form-group">
            <label>Email:</label>
            <input v-model="formData.email" type="email" required />
          </div>
          <div class="form-group">
            <label>Status:</label>
            <select v-model="formData.status">
              <option>active</option>
              <option>inactive</option>
              <option>pending</option>
            </select>
          </div>
          <div class="form-group">
            <label>Value:</label>
            <input v-model.number="formData.value" type="number" />
          </div>
          <button type="submit" class="btn-primary">Save</button>
          <button type="button" @click="closeModal" class="btn-secondary">Cancel</button>
        </form>
      </template>
    </modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import Modal from './Modal.vue';

interface DataRecord {
  id: string;
  name: string;
  email: string;
  status: 'active' | 'inactive' | 'pending';
  value: number;
  createdAt: Date;
}

defineProps({
  title: {
    type: String,
    default: 'Data Table'
  }
});

const records = ref<DataRecord[]>([]);
const searchQuery = ref('');
const showActive = ref(true);
const sortBy = ref('name');
const currentPage = ref(1);
const itemsPerPage = ref(10);
const showEditModal = ref(false);
const editingItem = ref<DataRecord | null>(null);
const selectedId = ref<string | null>(null);
const formData = ref({ name: '', email: '', status: 'active' as const, value: 0 });

const filteredAndSorted = computed(() => {
  let filtered = records.value.filter(item => {
    const matchesSearch =
      item.name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      item.email.toLowerCase().includes(searchQuery.value.toLowerCase());
    const matchesStatus = !showActive.value || item.status === 'active';
    return matchesSearch && matchesStatus;
  });

  filtered.sort((a, b) => {
    if (sortBy.value === 'name') return a.name.localeCompare(b.name);
    if (sortBy.value === 'date') return b.createdAt.getTime() - a.createdAt.getTime();
    if (sortBy.value === 'value') return b.value - a.value;
    return 0;
  });

  return filtered.slice(
    (currentPage.value - 1) * itemsPerPage.value,
    currentPage.value * itemsPerPage.value
  );
});

const totalPages = computed(() =>
  Math.ceil(
    records.value.filter(item =>
      !showActive.value || item.status === 'active'
    ).length / itemsPerPage.value
  )
);

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
};

const formatDate = (date: Date) => {
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }).format(date);
};

const addNewRow = () => {
  editingItem.value = null;
  formData.value = { name: '', email: '', status: 'active', value: 0 };
  showEditModal.value = true;
};

const editRow = (item: DataRecord) => {
  editingItem.value = item;
  formData.value = { ...item };
  showEditModal.value = true;
};

const deleteRow = (index: number) => {
  records.value.splice(index, 1);
};

const selectRow = (id: string) => {
  selectedId.value = selectedId.value === id ? null : id;
};

const saveChanges = () => {
  if (editingItem.value) {
    const index = records.value.findIndex(r => r.id === editingItem.value!.id);
    if (index !== -1) {
      records.value[index] = { ...editingItem.value, ...formData.value };
    }
  } else {
    records.value.push({
      id: Math.random().toString(36).substr(2, 9),
      ...formData.value,
      createdAt: new Date()
    });
  }
  closeModal();
};

const closeModal = () => {
  showEditModal.value = false;
  editingItem.value = null;
};

const handleSort = () => {
  currentPage.value = 1;
};

const nextPage = () => {
  if (currentPage.value < totalPages.value) currentPage.value++;
};

const previousPage = () => {
  if (currentPage.value > 1) currentPage.value--;
};

onMounted(() => {
  records.value = [
    { id: '1', name: 'Alice Johnson', email: 'alice@example.com', status: 'active', value: 1500, createdAt: new Date('2024-01-15') },
    { id: '2', name: 'Bob Smith', email: 'bob@example.com', status: 'pending', value: 2000, createdAt: new Date('2024-01-20') },
    { id: '3', name: 'Carol White', email: 'carol@example.com', status: 'active', value: 1200, createdAt: new Date('2024-02-01') }
  ];
});
</script>

<style scoped>
.data-table-container {
  padding: 20px;
  font-family: system-ui, -apple-system, sans-serif;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.controls {
  display: flex;
  gap: 10px;
}

.search-input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.filters {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.data-table thead {
  background-color: #f5f5f5;
}

.data-table th {
  padding: 12px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #ddd;
}

.data-table td {
  padding: 12px;
  border-bottom: 1px solid #eee;
}

.data-table tbody tr:hover {
  background-color: #fafafa;
}

.row-highlight {
  background-color: #e8f4f8;
}

.badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}

.badge-active {
  background-color: #d4edda;
  color: #155724;
}

.badge-inactive {
  background-color: #f8d7da;
  color: #721c24;
}

.badge-pending {
  background-color: #fff3cd;
  color: #856404;
}

.btn-sm {
  padding: 4px 8px;
  margin-right: 4px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.btn-danger {
  background-color: #dc3545;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-top: 20px;
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: #999;
}

.btn-primary,
.btn-secondary {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: 600;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}
</style>
