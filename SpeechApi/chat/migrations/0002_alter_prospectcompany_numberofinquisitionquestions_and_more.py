# Generated by Django 4.1.1 on 2023-04-20 17:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prospectcompany',
            name='numberOfInquisitionQuestions',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name='prospectcompany',
            name='numberOfObjectionQuestions',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
        migrations.AddField(
        model_name='questionssalesrepcompany',
        name='salesRepCompany_id',
        field=models.ForeignKey(default=None, on_delete=models.deletion.CASCADE, to='chat.ProspectCompany'),
        preserve_default=False,
    ),
    ]
