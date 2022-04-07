package xyz.laur.duels;

import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.block.BlockState;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityShootBowEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.inventory.meta.LeatherArmorMeta;
import org.bukkit.plugin.Plugin;
import org.bukkit.projectiles.ProjectileSource;
import org.bukkit.util.Vector;

import java.util.*;

import static org.bukkit.event.entity.EntityDamageEvent.DamageCause.FALL;

public class BridgeGameSession implements Listener, GameSession {
    public static final long MAX_DURATION = 3600; // 3 minutes

    private GameState state = GameState.WAITING_FOR_PLAYERS;
    private final Map<Block, BlockState> changedBlocks = new HashMap<>();
    private final List<Player> players = new ArrayList<>();
    private final Map<Player, Team> playerTeams = new HashMap<>();
    private final Map<Player, Integer> points = new HashMap<>();
    private final Map<Player, Location> lastLocations = new HashMap<>();
    private final BridgeMap map;
    private final World world;
    private final SessionManager sessionManager;
    private final InvisibilityManager invisibilityManager;
    private final SkinChanger skinChanger;
    private final Plugin plugin;
    private final double spawnDistanceFraction;

    private int movementMetadataTask;

    public BridgeGameSession(World world, SessionManager sessionManager, InvisibilityManager invisibilityManager,
                             SkinChanger skinChanger, Plugin plugin, BridgeMap map, double spawnDistanceFraction) {
        this.world = world;
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.skinChanger = skinChanger;
        this.plugin = plugin;
        this.map = map;
        this.spawnDistanceFraction = spawnDistanceFraction;

        plugin.getServer().getPluginManager().registerEvents(this, plugin);
    }

    public enum Team {
        BLUE, RED
    }

    public GameState getState() {
        return state;
    }

    public BridgeMap getMap() {
        return map;
    }

    @Override
    public String getMapName() {
        return map.getDisplayName();
    }

    public List<Player> getPlayers() {
        return players;
    }

    public void addPlayer(Player player) {
        if (state != GameState.WAITING_FOR_PLAYERS) throw new IllegalStateException("Session is in invalid state");
        if (players.size() >= 2) throw new IllegalStateException("Game already has too many players");
        if (hasPlayer(player)) throw new IllegalArgumentException("Player is already in this session");
        if (sessionManager.getPlayerSession(player) != null)
            throw new IllegalArgumentException("Player is already in a session");

        players.add(player);
        invisibilityManager.update();

        if (players.size() >= 2) {
            startGame();
        }
    }

    public void startGame() {
        if (state != GameState.WAITING_FOR_PLAYERS) throw new IllegalStateException("Session is in invalid state");
        if (players.size() != 2) throw new IllegalStateException("Need exactly 2 players to start a game");

        plugin.getLogger().info("Starting The Bridge duels game");

        state = GameState.PLAYING;

        List<Team> teams = new ArrayList<>(Arrays.asList(Team.BLUE, Team.RED));
        Collections.shuffle(teams);

        for (int i = 0; i < players.size(); i++) {
            Player player = players.get(i);
            Team playerTeam = teams.get(i);
            playerTeams.put(player, playerTeam);
            points.put(player, 0);
            spawnPlayer(player);
            player.sendMessage("Game started");
        }

        players.forEach(p -> lastLocations.put(p, p.getLocation()));
        movementMetadataTask = plugin.getServer().getScheduler().scheduleSyncRepeatingTask(plugin,
                this::movementMetadataTick, 0, 10);

        // Postpone changing skin because sometimes it makes players invisible if set
        // at the same time as players are teleported
        plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
            if (state == GameState.PLAYING) {
                players.forEach(skinChanger::changeSkin);
            }
        }, 100);

        plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
            if (state == GameState.PLAYING) {
                endGame(null);
                plugin.getLogger().info("The Bridge duel game ended due to timeout");
            }
        }, MAX_DURATION);
    }

    protected void movementMetadataTick() {
        if (state != GameState.PLAYING) {
            throw new IllegalStateException("Game was not in playing state but movement tick was called");
        }

        for (Player player : players) {
            Location otherHole = playerTeams.get(player) == Team.BLUE ?
                    map.getRedHole().clone() : map.getBlueHole().clone();
            otherHole.setWorld(world);

            double distanceBefore = lastLocations.get(player).distance(otherHole);
            double distanceNow = player.getLocation().distance(otherHole);

            double change = Math.abs(distanceNow - distanceBefore);

            if (change > 0.01 && change < 5) {
                sendMetadata(player, "distance_change", distanceNow - distanceBefore);
            }

            lastLocations.put(player, player.getLocation());
        }
    }

    protected void spawnPlayer(Player player) {
        if (state != GameState.PLAYING) throw new IllegalStateException("Game is not in playing state");

        player.setFallDistance(0);
        player.setGameMode(GameMode.SURVIVAL);
        player.setFlying(false);

        Team playerTeam = playerTeams.get(player);

        if (playerTeam == Team.BLUE) {
            player.teleport(getEffectiveSpawn(map.getBlueSpawn()));
        } else {
            player.teleport(getEffectiveSpawn(map.getRedSpawn()));
        }

        plugin.getLogger().info("Player spawned");
        player.setHealth(player.getMaxHealth());
        giveItems(player, playerTeam);
    }

    protected Location getEffectiveSpawn(Location original) {
        Location center = getCenter();

        Vector toFullSpawn = original.toVector().subtract(center.toVector());

        Location spawn = center.clone().add(toFullSpawn.multiply(spawnDistanceFraction));
        spawn.setY(255);

        while (spawn.getBlock().getRelative(BlockFace.DOWN).getType().isTransparent()) {
            if (spawn.getY() < 0) {
                throw new RuntimeException("Did not found a block to spawn player on");
            }

            spawn.subtract(0, 1, 0);
        }

        spawn.setPitch(original.getPitch());
        spawn.setYaw(original.getYaw());

        return spawn;
    }

    protected Location getCenter() {
        Location blueSpawn = map.getBlueSpawn().clone();
        blueSpawn.setWorld(world);

        Location redSpawn = map.getRedSpawn().clone();
        redSpawn.setWorld(world);

        return blueSpawn.add(redSpawn).multiply(0.5);
    }

    public void endGame(Player winner) {
        if (state == GameState.ENDED) throw new IllegalStateException("Game has already ended");

        state = GameState.ENDED;

        PlayerQuitEvent.getHandlerList().unregister(this);
        EntityDamageByEntityEvent.getHandlerList().unregister(this);
        EntityDamageEvent.getHandlerList().unregister(this);
        EntityShootBowEvent.getHandlerList().unregister(this);
        PlayerMoveEvent.getHandlerList().unregister(this);
        BlockBreakEvent.getHandlerList().unregister(this);
        BlockPlaceEvent.getHandlerList().unregister(this);

        plugin.getServer().getScheduler().cancelTask(movementMetadataTask);

        for (Player player : players) {
            player.getInventory().clear();
        }

        if (winner == null) {
            for (Player player : players) {
                sendMetadata(player, "timeout", 1);
                player.sendMessage("You lost");
            }
        } else {
            sendMetadata(winner, "win", 1);
            winner.sendMessage("You won");

            Player loser = getOtherPlayer(winner);
            loser.sendMessage("You lost");
        }

        sessionManager.removeSession(this);
        invisibilityManager.update();

        for (Block block : changedBlocks.keySet()) {
            BlockState blockState = changedBlocks.get(block);

            block.setType(blockState.getType());
            //noinspection deprecation
            block.setData(blockState.getRawData());
        }
    }

    protected void endRound(Player winner) {
        if (state != GameState.PLAYING) throw new IllegalStateException("Game has already ended");

        if (winner == null) {
            for (Player player : players) {
                sendMetadata(player, "round_lost", 1);
            }
        } else {
            int newPoints = points.get(winner) + 1;
            points.put(winner, newPoints);

            sendMetadata(winner, "round_won", 1);

            Player loser = getOtherPlayer(winner);
            sendMetadata(loser, "round_lost", 1);

            if (newPoints >= 4) {
                endGame(winner);
            } else {
                for (Player player : players) {
                    spawnPlayer(player);
                }
            }
        }
    }

    public boolean hasPlayer(Player player) {
        return players.contains(player);
    }

    public Team getPlayerTeam(Player player) {
        return playerTeams.get(player);
    }

    protected void giveItems(Player player, Team team) {
        player.getInventory().clear();

        ItemStack sword = new ItemStack(Material.IRON_SWORD);
        makeUnbreakable(sword);

        ItemStack bow = new ItemStack(Material.BOW);
        makeUnbreakable(bow);

        ItemStack diamondPickaxe = new ItemStack(Material.DIAMOND_PICKAXE);
        diamondPickaxe.addEnchantment(Enchantment.DIG_SPEED, 2);
        makeUnbreakable(diamondPickaxe);

        byte clayData = team == Team.BLUE ? DyeColor.BLUE.getWoolData() : DyeColor.RED.getWoolData();
        @SuppressWarnings("deprecation")
        ItemStack clay = new ItemStack(Material.STAINED_CLAY, 64, (short) 0, clayData);

        ItemStack goldenApple = new ItemStack(Material.GOLDEN_APPLE, 8);

        player.getInventory().addItem(sword, bow, diamondPickaxe, clay, clay, goldenApple);

        player.getInventory().setItem(7, new ItemStack(Material.DIAMOND));
        player.getInventory().setItem(8, new ItemStack(Material.ARROW));

        player.getInventory().setHelmet(null);

        ItemStack chestplate = new ItemStack(Material.LEATHER_CHESTPLATE);
        setColor(chestplate, team);
        makeUnbreakable(chestplate);
        player.getInventory().setChestplate(chestplate);

        ItemStack leggings = new ItemStack(Material.LEATHER_LEGGINGS);
        setColor(leggings, team);
        makeUnbreakable(leggings);
        player.getInventory().setLeggings(leggings);

        ItemStack boots = new ItemStack(Material.LEATHER_BOOTS);
        setColor(boots, team);
        makeUnbreakable(boots);
        player.getInventory().setBoots(boots);
    }

    protected void makeUnbreakable(ItemStack item) {
        ItemMeta meta = item.getItemMeta();
        meta.spigot().setUnbreakable(true);
        item.setItemMeta(meta);
    }

    protected void setColor(ItemStack item, Team team) {
        LeatherArmorMeta meta = ((LeatherArmorMeta) item.getItemMeta());
        meta.setColor(team == Team.BLUE ? Color.BLUE : Color.RED);
        item.setItemMeta(meta);
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        if (!hasPlayer(event.getPlayer()) || state != GameState.PLAYING) return;

        Player winner = getOtherPlayer(event.getPlayer());
        endGame(winner);

        plugin.getLogger().info("Game ended due to one player quiting");
    }

    @EventHandler
    public void onMove(PlayerMoveEvent event) {
        Player player = event.getPlayer();
        if (!hasPlayer(player) || state != GameState.PLAYING) return;

        if (event.getTo().getY() < map.getMinY()) {
            plugin.getLogger().info("Player fell to void");
            sendMetadata(player, "fell_to_void", 1);
            spawnPlayer(player);
            return;
        }

        Block blockDown = event.getTo().getBlock().getRelative(BlockFace.DOWN);
        if (blockDown.getType() == Material.BARRIER) {
            Location ownHole = playerTeams.get(player) == Team.BLUE ?
                    map.getBlueHole().clone() : map.getRedHole().clone();
            ownHole.setWorld(world);
            double distanceToOwn = ownHole.distance(event.getTo());

            if (distanceToOwn < 10) {
                plugin.getLogger().info("Player fell to their own hole, respawning them");
                sendMetadata(player, "fell_to_own_hole", 1);
                spawnPlayer(player);
                return;
            }

            Location otherHole = playerTeams.get(player) == Team.BLUE ?
                    map.getRedHole().clone() : map.getBlueHole().clone();
            otherHole.setWorld(world);
            double distanceToOther = otherHole.distance(event.getTo());

            if (distanceToOther < 10) {
                plugin.getLogger().info("Player jumped to opponent's hole");
                endRound(player);
                return;
            }

            plugin.getLogger().warning("Player was on barrier but no hole was near");
        }
    }

    @EventHandler
    public void onBowShoot(EntityShootBowEvent event) {
        if (state != GameState.PLAYING) return;
        if (!(event.getEntity() instanceof Player)) return;

        Player shooter = (Player) event.getEntity();
        if (hasPlayer(shooter)) {
            sendMetadata(shooter, "shoot_arrow", event.getForce());

            plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
                if (state == GameState.PLAYING) {
                    shooter.getInventory().setItem(8, new ItemStack(Material.ARROW));
                }
            }, 70);
        }
    }

    @EventHandler
    public void onBlockBreak(BlockBreakEvent event) {
        if (!hasPlayer(event.getPlayer())) return;

        Block block = event.getBlock();
        if (state == GameState.PLAYING && block.getType() == Material.STAINED_CLAY) {
            if (!changedBlocks.containsKey(block)) {
                changedBlocks.put(block, block.getState());
            }
        } else {
            event.setCancelled(true);
        }
    }

    @EventHandler
    public void onBlockPlace(BlockPlaceEvent event) {
        if (!hasPlayer(event.getPlayer())) return;

        Block block = event.getBlock();

        Location blueHole = map.getBlueHole().clone();
        blueHole.setWorld(world);
        double distanceToBlueHole = block.getLocation().distance(blueHole);

        Location redHole = map.getRedHole().clone();
        redHole.setWorld(world);
        double distanceToRedHole = block.getLocation().distance(redHole);

        if (state == GameState.PLAYING && distanceToRedHole > 12 && distanceToBlueHole > 12) {
            if (!changedBlocks.containsKey(block)) {
                changedBlocks.put(block, event.getBlockReplacedState());
            }
        } else {
            event.setCancelled(true);
        }
    }

    @EventHandler
    public void onEntityDamage(EntityDamageEvent event) {
        if (state != GameState.PLAYING) return;
        if (!(event.getEntity() instanceof Player)) return;

        Player player = (Player) event.getEntity();

        if (hasPlayer(player) && event.getCause() == FALL) {
            event.setCancelled(true);
            return;
        }

        if (hasPlayer(player) && player.getHealth() - event.getFinalDamage() <= 0) {
            plugin.getLogger().info("Player died");
            event.setCancelled(true);
            spawnPlayer(player);
        }
    }

    @EventHandler
    public void onPlayerDamage(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player) || state != GameState.PLAYING) {
            return;
        }

        Player target = (Player) event.getEntity();

        if (!hasPlayer(target)) return;

        if (event.getDamager() instanceof Player) {
            Player attacker = (Player) event.getDamager();

            handleDamage(attacker, target, event.getFinalDamage());
        } else if (event.getDamager() instanceof Projectile) {
            ProjectileSource shooter = ((Projectile) event.getDamager()).getShooter();

            if (shooter instanceof Player) {
                Player attacker = (Player) shooter;

                if (hasPlayer(attacker)) {
                    handleDamage(attacker, target, event.getFinalDamage());
                } else {
                    plugin.getLogger().info("Player got damaged by projectile from a player that is not in their session");
                    event.setCancelled(true);
                }
            }
        }
    }

    private void handleDamage(Player attacker, Player target, double damage) {
        if (hasPlayer(attacker) && hasPlayer(target) && damage > 0.1) {
            sendMetadata(attacker, "hits_done", damage);
            sendMetadata(target, "hits_received", damage);
        }
    }

    protected void sendMetadata(Player player, String key, double value) {
        player.sendMessage(String.format("metadata:%s:%.5f", key, value));
    }

    protected Player getOtherPlayer(Player player) {
        for (Player p : players) {
            if (!p.equals(player)) {
                return p;
            }
        }

        throw new IllegalStateException("Could not find other player");
    }
}
