"""Other plotting methods """

# 1) Aggregated data plot
fig, ax = plt.subplots()
sns.lineplot(data=df, x="time", y="A", hue="Patient", ax=ax)
xticks = ax.get_xticks()
xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
          for tm in xticks]
ax.set_xticklabels(xticks, rotation=45)
fig.tight_layout()


# 2)
# sns.relplot(
#     data=df, x="time", y="A",
#     col="DeMortonDay",  row='Patient',
#     hue="Side", # style="event",
#     kind="line"
# )

