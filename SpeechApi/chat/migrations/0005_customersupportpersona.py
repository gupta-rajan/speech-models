# Generated by Django 4.1.1 on 2023-04-26 01:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0004_alter_prospectcompany_current_question_index'),
    ]

    operations = [
        migrations.CreateModel(
            name='CustomerSupportPersona',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=100, null=True)),
                ('city', models.CharField(blank=True, max_length=100, null=True)),
                ('issue', models.CharField(blank=True, max_length=100, null=True)),
                ('company', models.CharField(blank=True, max_length=100, null=True)),
                ('product', models.CharField(blank=True, max_length=100, null=True)),
                ('personaGenerated', models.TextField(blank=True, null=True)),
            ],
        ),
    ]