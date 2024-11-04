import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# Create bot instance with command prefix '!'
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Event: Bot is ready
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

# Command: Simple ping command
@bot.command()
async def ping(ctx):
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

# Command: Echo message
@bot.command()
async def echo(ctx, *, message):
    await ctx.send(message)

# Command: Server info
@bot.command()
async def serverinfo(ctx):
    server = ctx.guild
    member_count = len(server.members)
    embed = discord.Embed(title=f"{server.name} Info", color=0x00ff00)
    embed.add_field(name="Server ID", value=server.id, inline=True)
    embed.add_field(name="Member Count", value=member_count, inline=True)
    embed.add_field(name="Created On", value=server.created_at.strftime("%b %d, %Y"), inline=True)
    await ctx.send(embed=embed)


# ChatGPT command
@bot.command()
async def chat(ctx, *, message):
    try:
        # Let user know the bot is thinking
        async with ctx.typing():
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                messages=[
                    {"role": "user", "content": message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Get the response text
            reply = response.choices[0].message.content

            # Split long responses into chunks if needed
            if len(reply) > 2000:
                # Discord has a 2000 character limit
                chunks = [reply[i:i+2000] for i in range(0, len(reply), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(reply)
    
    except Exception as e:
        await ctx.send(f"Sorry, I encountered an error: {str(e)}")

# Error handling for the chat command
@chat.error
async def chat_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Please provide a message to chat about! Usage: !chat <your message>")

# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found!")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have permission to use this command!")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

# Run the bot with your token
bot.run(os.getenv('DISCORD_TOKEN'))