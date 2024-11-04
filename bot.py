import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
import socket

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# Enable ALL intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return
    
    # ONLY handle direct messages that aren't commands
    if isinstance(message.channel, discord.DMChannel) and not message.content.startswith('!'):
        try:
            async with message.channel.typing():
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": message.content}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                reply = response.choices[0].message.content

                if len(reply) > 2000:
                    chunks = [reply[i:i+2000] for i in range(0, len(reply), 2000)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(reply)
        except Exception as e:
            await message.channel.send(f"Sorry, I encountered an error: {str(e)}")
    
    # For DM commands and all server messages, just process commands normally
    await bot.process_commands(message)

@bot.command()
async def ping(ctx):
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

@bot.command()
async def echo(ctx, *, message):
    await ctx.send(message)

@bot.command()
async def serverinfo(ctx):
    server = ctx.guild
    member_count = len(server.members)
    embed = discord.Embed(title=f"{server.name} Info", color=0x00ff00)
    embed.add_field(name="Server ID", value=server.id, inline=True)
    embed.add_field(name="Member Count", value=member_count, inline=True)
    embed.add_field(name="Created On", value=server.created_at.strftime("%b %d, %Y"), inline=True)
    await ctx.send(embed=embed)

@bot.command()
async def chat(ctx, *, message):
    try:
        async with ctx.typing():
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content

            if len(reply) > 2000:
                chunks = [reply[i:i+2000] for i in range(0, len(reply), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(reply)
    except Exception as e:
        await ctx.send(f"Sorry, I encountered an error: {str(e)}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found!")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have permission to use this command!")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

if __name__ == "__main__":
    try:
        bot.run(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        print(f"Failed to start bot: {str(e)}")
    