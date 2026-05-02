"""
Django views with class-based views, ORM queries, form handling.
"""
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, HttpResponseForbidden
from django.db.models import Q, F, Count, Prefetch
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.forms import ModelForm
from .models import Article, Comment, Tag, UserProfile


class ArticleListView(LoginRequiredMixin, ListView):
    model = Article
    template_name = 'articles/article_list.html'
    context_object_name = 'articles'
    paginate_by = 20

    def get_queryset(self):
        queryset = Article.objects.filter(
            published=True
        ).select_related(
            'author',
            'author__profile'
        ).prefetch_related(
            'tags',
            Prefetch('comment_set', queryset=Comment.objects.filter(approved=True))
        ).annotate(
            comment_count=Count('comment', filter=Q(comment__approved=True))
        ).order_by('-published_at')

        search = self.request.GET.get('q')
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) | Q(content__icontains=search)
            )

        tag = self.request.GET.get('tag')
        if tag:
            queryset = queryset.filter(tags__slug=tag)

        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['trending_tags'] = Tag.objects.annotate(
            usage=Count('article')
        ).order_by('-usage')[:5]
        context['search_query'] = self.request.GET.get('q', '')
        return context


class ArticleDetailView(DetailView):
    model = Article
    template_name = 'articles/article_detail.html'
    context_object_name = 'article'
    slug_field = 'slug'
    slug_url_kwarg = 'slug'

    def get_queryset(self):
        return Article.objects.select_related(
            'author', 'author__profile'
        ).prefetch_related('tags')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comments'] = self.object.comment_set.filter(
            approved=True
        ).select_related('author').order_by('created_at')
        context['related'] = Article.objects.filter(
            tags__in=self.object.tags.all(),
            published=True
        ).exclude(pk=self.object.pk).distinct()[:3]
        return context


class CommentForm(ModelForm):
    class Meta:
        model = Comment
        fields = ['content']


@login_required
@require_http_methods(['POST'])
def post_comment(request, slug):
    article = get_object_or_404(Article, slug=slug, published=True)
    form = CommentForm(request.POST)

    if form.is_valid():
        comment = form.save(commit=False)
        comment.article = article
        comment.author = request.user
        comment.approved = False
        comment.save()
        return JsonResponse({'status': 'success', 'message': 'Comment pending review'})

    return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)


class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    fields = ['title', 'content', 'excerpt', 'tags']
    template_name = 'articles/article_form.html'

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


class ArticleUpdateView(LoginRequiredMixin, UpdateView):
    model = Article
    fields = ['title', 'content', 'excerpt', 'tags', 'published']
    template_name = 'articles/article_form.html'

    def get_queryset(self):
        return Article.objects.filter(author=self.request.user)

    def get(self, request, *args, **kwargs):
        article = self.get_object()
        if article.author != request.user:
            return HttpResponseForbidden("Not your article")
        return super().get(request, *args, **kwargs)


@login_required
def user_dashboard(request):
    profile = get_object_or_404(UserProfile, user=request.user)
    articles = Article.objects.filter(
        author=request.user
    ).annotate(
        views=F('view_count'),
        comments=Count('comment')
    ).order_by('-created_at')[:10]

    stats = {
        'total_articles': articles.count(),
        'total_comments': Comment.objects.filter(
            article__author=request.user
        ).count(),
        'profile': profile,
        'recent_articles': articles,
    }
    return render(request, 'dashboard.html', stats)
