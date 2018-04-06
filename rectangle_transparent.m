function ax = rectangle_transparent(pos, colour)

    ax = rectangle('Position', pos);
    ax.LineStyle = '--';
    ax.EdgeColor = [colour];
    ax.FaceColor = [colour,0.2];

end

