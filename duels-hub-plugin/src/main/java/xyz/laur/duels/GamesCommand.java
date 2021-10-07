package xyz.laur.duels;

import org.bukkit.ChatColor;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.HumanEntity;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class GamesCommand implements CommandExecutor {
    private final SessionManager sessionManager;

    public GamesCommand(SessionManager sessionManager) {
        this.sessionManager = sessionManager;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        Map<String, GameSession> sessions = sessionManager.getAll();

        if (sessions.isEmpty()) {
            sender.sendMessage(ChatColor.YELLOW + "No active games");
            return true;
        }

        ChatColor lastColor = ChatColor.YELLOW;

        for (String name : sessions.keySet()) {
            GameSession session = sessions.get(name);

            ChatColor nextColor = lastColor == ChatColor.YELLOW ? ChatColor.GOLD : ChatColor.YELLOW;

            String format = nextColor + "%s: [%s], %s, %s";

            List<String> playerNames = session.getPlayers()
                .stream()
                .map(HumanEntity::getName)
                .collect(Collectors.toList());

            String playerString = String.join(", ", playerNames);
            String stateString = formatGameState(session.getState());
            String mapName = session.getMap().getDisplayName();

            sender.sendMessage(String.format(format, name, playerString, stateString, mapName));

            lastColor = nextColor;
        }

        return true;
    }

    private String formatGameState(GameSession.State state) {
        switch (state) {
            case WAITING_FOR_PLAYERS:
                return "Waiting for Players";
            case PLAYING:
                return "Playing";
            case ENDED:
                return "Ended";
        }

        throw new IllegalArgumentException("Invalid state");
    }
}
