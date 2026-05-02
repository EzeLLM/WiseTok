package com.example.wisetok.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;

/**
 * Main activity for tokenizer training interface.
 * Demonstrates Android lifecycle, RecyclerView with adapters, ViewModel, and LiveData.
 */
public class TokenizerTrainingActivity extends AppCompatActivity {

    private TokenizerViewModel viewModel;
    private TokenListAdapter adapter;
    private RecyclerView recyclerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tokenizer_training);

        // Initialize ViewModel
        viewModel = new ViewModelProvider(this).get(TokenizerViewModel.class);

        // Setup RecyclerView
        recyclerView = findViewById(R.id.recycler_view_tokenizers);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        adapter = new TokenListAdapter(this::onTokenizerClick);
        recyclerView.setAdapter(adapter);

        // Observe LiveData
        viewModel.getTokenizers().observe(this, new Observer<List<TokenizerItem>>() {
            @Override
            public void onChanged(List<TokenizerItem> tokenizers) {
                if (tokenizers != null) {
                    adapter.submitList(new ArrayList<>(tokenizers));
                }
            }
        });

        viewModel.getLoadingState().observe(this, isLoading -> {
            findViewById(R.id.progress_bar).setVisibility(isLoading ? View.VISIBLE : View.GONE);
        });

        viewModel.getErrorState().observe(this, errorMsg -> {
            if (errorMsg != null && !errorMsg.isEmpty()) {
                showErrorDialog(errorMsg);
            }
        });

        // Load initial data
        viewModel.loadTokenizers();
    }

    @Override
    protected void onStart() {
        super.onStart();
        // Resume any paused operations
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Handle any foreground-specific logic
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Pause ongoing operations
    }

    @Override
    protected void onStop() {
        super.onStop();
        // Clean up resources
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Final cleanup
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putInt("recycler_position", ((LinearLayoutManager) recyclerView.getLayoutManager())
                .findFirstCompletelyVisibleItemPosition());
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        int position = savedInstanceState.getInt("recycler_position", 0);
        recyclerView.scrollToPosition(position);
    }

    private void onTokenizerClick(TokenizerItem item) {
        Intent intent = new Intent(this, TokenizerDetailActivity.class);
        intent.putExtra("tokenizer_id", item.getId());
        startActivity(intent);
    }

    private void showErrorDialog(String message) {
        // Show error dialog implementation
    }
}

class TokenListAdapter extends RecyclerView.Adapter<TokenListAdapter.TokenViewHolder> {

    private List<TokenizerItem> items = new ArrayList<>();
    private final OnTokenizerClickListener listener;

    public TokenListAdapter(OnTokenizerClickListener listener) {
        this.listener = listener;
    }

    public void submitList(List<TokenizerItem> newItems) {
        this.items = newItems;
        notifyDataSetChanged();
    }

    @Override
    public TokenViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = android.view.LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_tokenizer, parent, false);
        return new TokenViewHolder(view);
    }

    @Override
    public void onBindViewHolder(TokenViewHolder holder, int position) {
        TokenizerItem item = items.get(position);
        holder.bind(item, listener);
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class TokenViewHolder extends RecyclerView.ViewHolder {
        private final android.widget.TextView nameText;
        private final android.widget.TextView mergesText;
        private final android.widget.Button trainButton;

        TokenViewHolder(View itemView) {
            super(itemView);
            this.nameText = itemView.findViewById(R.id.text_tokenizer_name);
            this.mergesText = itemView.findViewById(R.id.text_merge_count);
            this.trainButton = itemView.findViewById(R.id.button_train);
        }

        void bind(TokenizerItem item, OnTokenizerClickListener listener) {
            nameText.setText(item.getName());
            mergesText.setText("Merges: " + item.getMergeCount());
            itemView.setOnClickListener(v -> listener.onTokenizerClick(item));
            trainButton.setOnClickListener(v -> listener.onTokenizerClick(item));
        }
    }

    interface OnTokenizerClickListener {
        void onTokenizerClick(TokenizerItem item);
    }
}

class TokenizerViewModel extends androidx.lifecycle.ViewModel {
    private final androidx.lifecycle.MutableLiveData<List<TokenizerItem>> tokenizers =
            new androidx.lifecycle.MutableLiveData<>();
    private final androidx.lifecycle.MutableLiveData<Boolean> loadingState =
            new androidx.lifecycle.MutableLiveData<>(false);
    private final androidx.lifecycle.MutableLiveData<String> errorState =
            new androidx.lifecycle.MutableLiveData<>();

    public LiveData<List<TokenizerItem>> getTokenizers() {
        return tokenizers;
    }

    public LiveData<Boolean> getLoadingState() {
        return loadingState;
    }

    public LiveData<String> getErrorState() {
        return errorState;
    }

    public void loadTokenizers() {
        loadingState.setValue(true);
        try {
            // Simulate data loading
            List<TokenizerItem> items = new ArrayList<>();
            items.add(new TokenizerItem(1L, "GPT-4 Pattern", 50000));
            items.add(new TokenizerItem(2L, "Custom Pattern", 30000));
            tokenizers.setValue(items);
            errorState.setValue(null);
        } catch (Exception e) {
            errorState.setValue("Failed to load tokenizers: " + e.getMessage());
        } finally {
            loadingState.setValue(false);
        }
    }

    public void deleteTokenizer(long id) {
        // Delete operation
    }
}

class TokenizerItem {
    private final Long id;
    private final String name;
    private final Integer mergeCount;

    public TokenizerItem(Long id, String name, Integer mergeCount) {
        this.id = id;
        this.name = name;
        this.mergeCount = mergeCount;
    }

    public Long getId() { return id; }
    public String getName() { return name; }
    public Integer getMergeCount() { return mergeCount; }
}

class TokenizerDetailActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tokenizer_detail);
        Long tokenizerId = getIntent().getLongExtra("tokenizer_id", -1);
    }
}
