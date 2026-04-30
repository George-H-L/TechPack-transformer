from django.contrib import admin
from .models import TechPack

@admin.register(TechPack)
class TechPackAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'created_at']
