function ax_area = area_transparent(x, y, colour)

    ax_area = area(x,y);
    ax_area.FaceAlpha = 0.2;
    ax_area.LineStyle = '--';
    ax_area.EdgeColor = colour;
    ax_area.FaceColor = colour;

end

