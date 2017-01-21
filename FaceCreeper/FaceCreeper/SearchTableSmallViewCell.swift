//
//  SearchTableSmallViewCell.swift
//  FaceCreeper
//
//  Created by Austin Han on 2017-01-21.
//  Copyright © 2017 Hackathon. All rights reserved.
//

import UIKit

class SearchTableSmallViewCell: UITableViewCell {
    @IBOutlet weak var profileImageView: UIImageView!

    @IBOutlet weak var percentageLabel: UILabel!
    @IBOutlet weak var nameLabel: UILabel!
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
