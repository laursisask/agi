package xyz.laur.duels;

import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;

public class BridgeJoinCommand implements CommandExecutor {
    private final SessionManager sessionManager;
    private final InvisibilityManager invisibilityManager;
    private final SkinChanger skinChanger;
    private final Plugin plugin;
    private final World world;

    public BridgeJoinCommand(SessionManager sessionManager, InvisibilityManager invisibilityManager,
                             SkinChanger skinChanger, Plugin plugin, World world) {
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.skinChanger = skinChanger;
        this.plugin = plugin;
        this.world = world;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage(ChatColor.RED + "Only players can join games");
            return true;
        }

        if (args.length != 3) {
            return false;
        }

        String sessionName = args[0];
        BridgeMap map = BridgeMap.fromName(args[1]);
        double spawnDistanceFraction = Double.parseDouble(args[2]);

        BridgeGameSession session = (BridgeGameSession) sessionManager.getByName(sessionName);

        if (session == null) {
            plugin.getLogger().info("Creating new bridge session");
            session = new BridgeGameSession(world, sessionManager, invisibilityManager, skinChanger, plugin, map,
                    spawnDistanceFraction);
            sessionManager.addSession(sessionName, session);
        }

        session.addPlayer((Player) sender);

        return true;
    }
}
